"""
Continual learning variant of TOFU capacity stress test.

Instead of training on all N authors at once, incrementally adds
AUTHORS_PER_STEP authors at a time to the same cartridge checkpoint.

Step 0:   Initialize cache from corpus text, train on first batch of authors.
Step t>0: Load cache from previous step's checkpoint, train on NEW authors only.

After all steps, evaluates each step's checkpoint on ALL authors seen up to
that point (measures retention of old knowledge + acquisition of new).

Usage:
    NUM_AUTHORS=10 NUM_TOKENS=64 AUTHORS_PER_STEP=2 python stress_test/tofu/tofu_train_continual.py
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import pydrantic
from pydrantic.variables import FormatStringVariable
from transformers import AutoTokenizer
from tqdm import tqdm

from cartridges.initialization import KVFromText, KVFromLocalPath
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.datasets import TrainDataset, DataSource
from cartridges.data.tofu.evals import TOFUQAGenerateDataset
from cartridges.data.tofu.utils import (
    load_tofu_authors,
    authors_to_conversations,
    authors_to_corpus_text,
    rescore_tofu_conversations,
    save_corpus_to_tempfile,
    save_conversations_to_parquet,
)
from cartridges.utils.wandb import WandBConfig


# --- Configuration from environment ---
os.environ["CARTRIDGES_WANDB_PROJECT"] = "cartridges"
os.environ["CARTRIDGES_WANDB_ENTITY"] = "izaaz-personal"

NUM_AUTHORS = int(os.environ.get("NUM_AUTHORS", "10"))
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "64"))
AUTHORS_PER_STEP = int(os.environ.get("AUTHORS_PER_STEP", "2"))
MODEL = os.environ.get("MODEL", "llama")
TARGETS = os.environ.get("TARGETS", "tokens")  # "logits" or "tokens"
output_dir = os.environ.get("CARTRIDGES_OUTPUT_DIR", ".")

# --- Model selection ---
if MODEL == "llama":
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
    from cartridges.models import HFModelConfig
    model_config = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
elif MODEL == "llama3b":
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
    from cartridges.models import HFModelConfig
    model_config = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
elif MODEL == "olmo":
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
    from cartridges.models import HFModelConfig
    model_config = HFModelConfig(
        pretrained_model_name_or_path="allenai/OLMo-3-7B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
elif MODEL == "qwen":
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    from cartridges.models import HFModelConfig
    model_config = HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}. Use 'llama', 'llama3b', 'olmo', or 'qwen'.")


def prepare_step_data(
    step_authors,
    cartridge_name: str,
    rescore_model=None,
    rescore_tokenizer=None,
):
    """
    Prepare training data for a single step's NEW authors only.

    Returns (train_data_path, num_conversations).
    """
    conversations = authors_to_conversations(step_authors)

    if TARGETS == "logits":
        assert rescore_model is not None and rescore_tokenizer is not None
        # For rescoring we need a corpus context — use just the new authors
        corpus_text = authors_to_corpus_text(step_authors)
        corpus_ntok = len(rescore_tokenizer.encode(corpus_text))
        rescore_seq_length = max(2048, corpus_ntok + 512)
        print(f"  Rescoring {len(conversations)} conversations for {cartridge_name}...")
        conversations = rescore_tofu_conversations(
            conversations, rescore_model, rescore_tokenizer,
            top_k=20, device="cuda",
            corpus_text=corpus_text,
            packed_seq_length=rescore_seq_length,
        )

    train_path = os.path.join(output_dir, f"tofu_continual_{cartridge_name}.parquet")
    save_conversations_to_parquet(conversations, train_path)

    return train_path, len(conversations)


def evaluate_cache(
    cache,
    model,
    tokenizer,
    num_authors: int,
    author_offset: int = 0,
    label: str = "eval",
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Run generation eval over the specified authors using *cache*."""
    from cartridges.generation import flex_generate

    eval_dataset = TOFUQAGenerateDataset(
        config=TOFUQAGenerateDataset.Config(
            num_authors=num_authors,
            author_offset=author_offset,
            seed=42,
        ),
        tokenizer=tokenizer,
        seed=42,
    )

    print(f"  [{label}] Evaluating {len(eval_dataset)} questions...")

    results = []
    for batch_start in tqdm(
        range(0, len(eval_dataset), batch_size),
        desc=f"Generating ({label})",
        leave=False,
    ):
        elements = [
            (seq_id, eval_dataset[idx])
            for seq_id, idx in enumerate(
                range(batch_start, min(batch_start + batch_size, len(eval_dataset)))
            )
        ]
        if not elements:
            continue

        input_ids = torch.cat([elem.input_ids[0] for _, elem in elements]).to(device)
        seq_ids = torch.cat(
            [
                torch.full((elem.input_ids.shape[1],), seq_id, dtype=torch.long, device=device)
                for seq_id, elem in elements
            ]
        )
        position_ids = torch.cat(
            [torch.arange(elem.input_ids.shape[1], device=device) for _, elem in elements]
        )

        pred_ids = flex_generate(
            input_ids=input_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
            cache=cache,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            show_progress=False,
        )

        elements_map = {seq_id: elem for seq_id, elem in elements}

        for seq_id, curr_pred_ids in pred_ids.items():
            element = elements_map[seq_id]
            pred = tokenizer.decode(curr_pred_ids, skip_special_tokens=True)
            orig_idx = batch_start + seq_id

            score_val, extras = eval_dataset.score(
                pred=pred,
                answer=element.answer,
                convo_id=element.convo_id,
            )

            score_dict = score_val if isinstance(score_val, dict) else {"score": score_val}
            results.append({
                "index": orig_idx,
                "author_index": element.metadata["author_index"],
                "qa_index": element.metadata["qa_index"],
                "convo_id": element.convo_id,
                "prompt": element.prompt,
                "answer": element.answer,
                "pred": pred,
                **score_dict,
                **extras,
            })

    cache.clear()
    return pd.DataFrame(results)


def evaluate_from_path(
    cache_path: str,
    model,
    tokenizer,
    num_authors: int,
    author_offset: int = 0,
    label: str = "cartridge",
    device: str = "cuda",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Load a cache from disk and evaluate it."""
    from cartridges.cache import TrainableCache

    cache = TrainableCache.from_pretrained(cache_path, device=device).to(device)
    return evaluate_cache(
        cache=cache,
        model=model,
        tokenizer=tokenizer,
        num_authors=num_authors,
        author_offset=author_offset,
        label=label,
        device=device,
        batch_size=batch_size,
    )


def _format_matrix_row(
    label: str,
    row_data: Dict,
    num_steps: int,
    row_label_width: int,
    col_width: int,
) -> str:
    """Format a single row for the forgetting matrix."""
    row = f"{label:<{row_label_width}}"
    for s in range(num_steps):
        key = f"group_{s}"
        if key in row_data:
            row += f"{row_data[key]:>{col_width}.4f}"
        else:
            row += f"{'x':>{col_width}}"
    if "all_seen" in row_data:
        row += f"{row_data['all_seen']:>{col_width}.4f}"
    else:
        row += f"{'x':>{col_width}}"
    return row


def print_matrix_summary(
    score_matrix: Dict[int, Dict],
    step_info: list,
    score_col: str = "rouge_l",
    joint_scores: Optional[Dict[int, Dict]] = None,
):
    """
    Print a forgetting-matrix style summary.

    Rows   = checkpoint (which step produced this cache)
    Cols   = author group per step + "All Seen"
    Cells  = mean score (or 'x' if those authors weren't seen yet)

    If joint_scores is provided, each continual row is followed by
    the corresponding joint-training baseline for comparison.
    """
    if not score_matrix:
        print("\n  No evaluation results to display.")
        return

    num_steps = max(score_matrix.keys()) + 1
    col_width = 14
    row_label_width = 14

    # Build column headers: "Step 0 (0-1)", "Step 1 (2-3)", ..., "All Seen"
    col_headers = []
    for s in range(num_steps):
        _, _, si, ei = step_info[s]
        col_headers.append(f"Step {s} ({si}-{ei-1})")
    col_headers.append("All Seen")

    header = f"{'Checkpoint':<{row_label_width}}" + "".join(
        f"{h:>{col_width}}" for h in col_headers
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print("  CONTINUAL LEARNING — FORGETTING MATRIX")
    print(f"  {NUM_AUTHORS} authors | R={NUM_TOKENS} tokens | "
          f"{AUTHORS_PER_STEP} authors/step | targets={TARGETS} | model={MODEL}")
    print(f"  Score: {score_col}")
    if joint_scores:
        print("  (Joint = trained from scratch on all seen authors at that step)")
    print(separator)
    print(header)
    print(separator)

    for ckpt_step in sorted(score_matrix.keys()):
        # Continual learning row
        print(_format_matrix_row(
            f"Step {ckpt_step}", score_matrix[ckpt_step],
            num_steps, row_label_width, col_width,
        ))
        # Joint baseline row (if available)
        if joint_scores and ckpt_step in joint_scores:
            print(_format_matrix_row(
                f"  Joint {ckpt_step}", joint_scores[ckpt_step],
                num_steps, row_label_width, col_width,
            ))

    print(separator)


def main():
    """
    Continual learning loop:
    1. Load all N authors.
    2. Step 0: init cache from corpus text, train on first batch of authors.
    3. Step t>0: load cache from previous checkpoint, train on NEW authors only.
    4. After all steps: evaluate each checkpoint on ALL the authors it has
       seen so far (retention + new knowledge).
    """
    all_authors = load_tofu_authors(num_authors=NUM_AUTHORS, seed=42)
    num_steps = (NUM_AUTHORS + AUTHORS_PER_STEP - 1) // AUTHORS_PER_STEP

    print(f"Continual learning: {NUM_AUTHORS} authors in {num_steps} steps "
          f"({AUTHORS_PER_STEP} authors/step), R={NUM_TOKENS}, targets={TARGETS}")

    # --- Optional: load model once for logit rescoring ---
    rescore_model, rescore_tokenizer = None, None
    if TARGETS == "logits":
        print("\nLoading base model for rescoring...")
        rescore_model = model_config.instantiate().to("cuda").to(torch.bfloat16)
        rescore_tokenizer = AutoTokenizer.from_pretrained(
            model_config.pretrained_model_name_or_path
        )

    # --- Prepare training data for each step (new authors only) ---
    print("\n=== Preparing step data ===")
    step_info = []  # [(train_path, n_convos, start_idx, end_idx)]
    for step in range(num_steps):
        start_idx = step * AUTHORS_PER_STEP
        end_idx = min(start_idx + AUTHORS_PER_STEP, NUM_AUTHORS)
        step_authors = all_authors[start_idx:end_idx]

        train_path, n_convos = prepare_step_data(
            step_authors,
            cartridge_name=f"step{step}_a{start_idx}-{end_idx - 1}",
            rescore_model=rescore_model,
            rescore_tokenizer=rescore_tokenizer,
        )
        step_info.append((train_path, n_convos, start_idx, end_idx))
        print(f"  Step {step}: authors {start_idx}-{end_idx - 1} "
              f"({n_convos} conversations)")

    # Free rescoring model before training
    if rescore_model is not None:
        del rescore_model
        torch.cuda.empty_cache()

    # --- Training loop with eval after each step ---
    prev_cache_path: Optional[str] = None
    step_configs: List[TrainConfig] = []

    # score_matrix[ckpt_step] = {"group_0": score, "group_1": score, ..., "all_seen": score}
    score_matrix: Dict[int, Dict] = {}

    # Load model/tokenizer once — used for eval after each step
    device = "cuda"
    eval_model = model_config.instantiate().to(device).to(torch.bfloat16)
    eval_tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name_or_path)

    for step in range(num_steps):
        train_path, n_convos, start_idx, end_idx = step_info[step]
        cumulative_end = end_idx  # total authors seen so far

        print(f"\n{'=' * 60}")
        print(f"=== Step {step + 1}/{num_steps}: "
              f"TRAIN on NEW authors {start_idx}-{end_idx - 1}, "
              f"EVAL on all seen 0-{cumulative_end - 1} ===")
        print(f"{'=' * 60}")

        # --- Choose cache initializer ---
        if step == 0:
            # First step: initialize from corpus text of the first batch
            corpus_path = save_corpus_to_tempfile(all_authors[:end_idx])
            kv_init = KVFromText.Config(
                text_source=corpus_path,
                max_tokens=NUM_TOKENS,
            )
            print(f"  Init: KVFromText (corpus of authors 0-{end_idx - 1})")
        else:
            # Subsequent steps: load previous checkpoint
            assert prev_cache_path is not None, (
                f"Step {step}: expected cache from previous step but none found"
            )
            kv_init = KVFromLocalPath.Config(path=prev_cache_path)
            print(f"  Init: KVFromLocalPath ({prev_cache_path})")

        config = TrainConfig(
            model=model_config,
            kv_cache_initializer=kv_init,
            lr=5e-3,
            epochs=10,
            global_batch_size=1,
            dataset=TrainDataset.Config(
                data_sources=[DataSource(path=train_path, type="local")],
                targets=TARGETS,
                top_k_logits=20,
                packed_seq_length=2048,
                packing_mode="pad",
            ),
            generate_eval_every_n_steps=10,
            generate_evals=[
                GenerationEvalConfig(
                    dataset=TOFUQAGenerateDataset.Config(
                        num_authors=cumulative_end,
                        seed=42,
                    ),
                    name_for_wandb=f"tofu_continual_step{step}_all_n{cumulative_end}",
                    generate_max_new_tokens=256,
                    batch_size=min(32, cumulative_end * 20),
                    temperature=0.0,
                )
            ],
            distributed_backend="gloo",
            save_every_n_steps=256,
            save_after_training=True,
            wandb=WandBConfig(
                tags=["tofu", "continual", f"step{step}", f"n{cumulative_end}",
                      f"r{NUM_TOKENS}", f"targets_{TARGETS}"],
                notes=(f"step={step}, train_authors={start_idx}-{end_idx-1}, "
                       f"all_seen=0-{cumulative_end-1}"),
            ),
            output_dir=output_dir,
            name=FormatStringVariable(
                f"tofu_continual_step{step}_n{cumulative_end}_r{NUM_TOKENS}_lr{{lr}}"
            ),
        )

        pydrantic.main(config)
        step_configs.append(config)

        # Track checkpoint for next step
        prev_cache_path = os.path.join(config.run_dir, "cache_last.pt")
        if not os.path.exists(prev_cache_path):
            print(f"  WARNING: cache_last.pt not found at {prev_cache_path}")
            prev_cache_path = None

        print(f"  Step {step + 1} training complete. Cache: {prev_cache_path}")

        # --- Evaluate RIGHT AFTER this step ---
        cache_path = os.path.join(config.run_dir, "cache_last.pt")
        if not os.path.exists(cache_path):
            print(f"  Step {step}: cache not found at {cache_path}, skipping eval.")
            continue

        score_matrix[step] = {}

        # Evaluate on each individual author group seen so far
        for prev_step in range(step + 1):
            _, _, gs, ge = step_info[prev_step]
            group_count = ge - gs
            print(f"  --- Step {step} eval: group {prev_step} (authors {gs}-{ge-1}) ---")
            df = evaluate_from_path(
                cache_path=cache_path,
                model=eval_model, tokenizer=eval_tokenizer,
                num_authors=group_count,
                author_offset=gs,
                label=f"step{step}-group{prev_step}",
                device=device,
                batch_size=min(32, group_count * 20),
            )
            # Extract mean score (prefer rouge_l)
            if "rouge_l" in df.columns:
                score_matrix[step][f"group_{prev_step}"] = df["rouge_l"].mean()
            elif "score" in df.columns:
                score_matrix[step][f"group_{prev_step}"] = df["score"].mean()

        # Evaluate on ALL seen authors combined
        print(f"  --- Step {step} eval: all seen authors 0-{cumulative_end - 1} ---")
        df_all = evaluate_from_path(
            cache_path=cache_path,
            model=eval_model, tokenizer=eval_tokenizer,
            num_authors=cumulative_end,
            label=f"step{step}-all-seen",
            device=device,
            batch_size=min(32, cumulative_end * 20),
        )
        if "rouge_l" in df_all.columns:
            score_matrix[step]["all_seen"] = df_all["rouge_l"].mean()
        elif "score" in df_all.columns:
            score_matrix[step]["all_seen"] = df_all["score"].mean()


    # --- Joint baseline: train from scratch on ALL cumulative authors per step ---
    print("\n" + "=" * 60)
    print("=== Joint baselines: training from scratch at each cumulative size ===")
    print("=" * 60)

    joint_scores: Dict[int, Dict] = {}

    for step in range(num_steps):
        _, _, start_idx, end_idx = step_info[step]
        cumulative_end = end_idx
        cumulative_authors = all_authors[:cumulative_end]

        print(f"\n--- Joint baseline for step {step}: "
              f"all {cumulative_end} authors from scratch ---")

        # Prepare training data: ALL cumulative authors
        joint_conversations = authors_to_conversations(cumulative_authors)
        joint_train_path = os.path.join(
            output_dir, f"tofu_continual_joint_n{cumulative_end}.parquet"
        )
        save_conversations_to_parquet(joint_conversations, joint_train_path)
        joint_corpus_path = save_corpus_to_tempfile(cumulative_authors)

        joint_config = TrainConfig(
            model=model_config,
            kv_cache_initializer=KVFromText.Config(
                text_source=joint_corpus_path,
                max_tokens=NUM_TOKENS,
            ),
            lr=5e-3,
            epochs=15,
            global_batch_size=1,
            dataset=TrainDataset.Config(
                data_sources=[DataSource(path=joint_train_path, type="local")],
                targets=TARGETS,
                top_k_logits=20,
                packed_seq_length=2048,
                packing_mode="pad",
            ),
            generate_eval_every_n_steps=10,
            generate_evals=[
                GenerationEvalConfig(
                    dataset=TOFUQAGenerateDataset.Config(
                        num_authors=cumulative_end,
                        seed=42,
                    ),
                    name_for_wandb=f"tofu_continual_joint_n{cumulative_end}",
                    generate_max_new_tokens=256,
                    batch_size=min(32, cumulative_end * 20),
                    temperature=0.0,
                )
            ],
            distributed_backend="gloo",
            save_every_n_steps=256,
            save_after_training=True,
            wandb=WandBConfig(
                tags=["tofu", "continual", "joint-baseline", f"n{cumulative_end}",
                      f"r{NUM_TOKENS}", f"targets_{TARGETS}"],
                notes=f"Joint baseline: all {cumulative_end} authors from scratch",
            ),
            output_dir=output_dir,
            name=FormatStringVariable(
                f"tofu_continual_joint_n{cumulative_end}_r{NUM_TOKENS}_lr{{lr}}"
            ),
        )

        pydrantic.main(joint_config)

        # Evaluate joint checkpoint
        joint_cache_path = os.path.join(joint_config.run_dir, "cache_last.pt")
        if not os.path.exists(joint_cache_path):
            print(f"  Joint step {step}: cache not found, skipping eval.")
            continue

        joint_scores[step] = {}

        for prev_step in range(step + 1):
            _, _, gs, ge = step_info[prev_step]
            group_count = ge - gs
            print(f"  --- Joint {step} eval: group {prev_step} (authors {gs}-{ge-1}) ---")
            df = evaluate_from_path(
                cache_path=joint_cache_path,
                model=eval_model, tokenizer=eval_tokenizer,
                num_authors=group_count,
                author_offset=gs,
                label=f"joint{step}-group{prev_step}",
                device=device,
                batch_size=min(32, group_count * 20),
            )
            if "rouge_l" in df.columns:
                joint_scores[step][f"group_{prev_step}"] = df["rouge_l"].mean()
            elif "score" in df.columns:
                joint_scores[step][f"group_{prev_step}"] = df["score"].mean()

        print(f"  --- Joint {step} eval: all seen 0-{cumulative_end-1} ---")
        df_all = evaluate_from_path(
            cache_path=joint_cache_path,
            model=eval_model, tokenizer=eval_tokenizer,
            num_authors=cumulative_end,
            label=f"joint{step}-all-seen",
            device=device,
            batch_size=min(32, cumulative_end * 20),
        )
        if "rouge_l" in df_all.columns:
            joint_scores[step]["all_seen"] = df_all["rouge_l"].mean()
        elif "score" in df_all.columns:
            joint_scores[step]["all_seen"] = df_all["score"].mean()

    del eval_model
    torch.cuda.empty_cache()

    # --- Final summary ---
    print("\n\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    score_col_name = "rouge_l"  # default
    print_matrix_summary(
        score_matrix, step_info,
        score_col=score_col_name,
        joint_scores=joint_scores,
    )


if __name__ == "__main__":
    main()
