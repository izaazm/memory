"""
Modular cartridge comparison for TOFU capacity stress test.

Splits N authors into 2 groups and trains separate cartridges (A and B).
After training, concatenates both cartridges' KV caches and evaluates
all questions against the composed cache (no routing).

Compares:
- Monolithic: 1 cartridge with all N authors at budget R
- Modular: 2 cartridges with N/2 authors each at budget R/2, then composed

Usage:
    NUM_AUTHORS=10 NUM_TOKENS=64 python stress_test/tofu/tofu_train_modular.py
"""
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import pydrantic
from pydrantic.variables import FormatStringVariable
from transformers import AutoTokenizer
from tqdm import tqdm

from cartridges.initialization import KVFromText
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
MODEL = os.environ.get("MODEL", "llama")
TARGETS = os.environ.get("TARGETS", "tokens")  # "logits" or "tokens"
output_dir = os.environ.get("CARTRIDGES_OUTPUT_DIR", ".")

# --- Model selection (same as tofu_train.py) ---
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


def prepare_data(authors, cartridge_name: str, rescore_model=None, rescore_tokenizer=None):
    """
    Prepare training data for a subset of authors.

    If TARGETS="logits", runs rescoring with the provided model/tokenizer.
    If TARGETS="tokens", just saves conversations directly (SFT).

    Returns (train_data_path, corpus_path, corpus_num_tokens).
    """
    conversations = authors_to_conversations(authors)
    corpus_text = authors_to_corpus_text(authors)
    tokenizer = rescore_tokenizer or AutoTokenizer.from_pretrained(
        model_config.pretrained_model_name_or_path
    )
    corpus_num_tokens = len(tokenizer.encode(corpus_text))

    if TARGETS == "logits":
        assert rescore_model is not None and rescore_tokenizer is not None, (
            "rescore_model and rescore_tokenizer required for targets='logits'"
        )
        rescore_seq_length = max(2048, corpus_num_tokens + 512)
        print(f"  Rescoring {len(conversations)} conversations for {cartridge_name}...")
        conversations = rescore_tofu_conversations(
            conversations, rescore_model, rescore_tokenizer,
            top_k=20, device="cuda",
            corpus_text=corpus_text,
            packed_seq_length=rescore_seq_length,
        )

    train_path = os.path.join(output_dir, f"tofu_modular_{cartridge_name}.parquet")
    save_conversations_to_parquet(conversations, train_path)
    corpus_path = save_corpus_to_tempfile(authors)

    return train_path, corpus_path, corpus_num_tokens


def make_train_config(
    train_path: str,
    corpus_path: str,
    corpus_num_tokens: int,
    cartridge_name: str,
    num_tokens: int,
    num_eval_authors: int,
    author_offset: int = 0,
    num_conversations: int = 1,
) -> TrainConfig:
    """Build a TrainConfig for a single cartridge."""
    return TrainConfig(
        model=model_config,
        kv_cache_initializer=KVFromText.Config(
            text_source=corpus_path,
            max_tokens=num_tokens,
        ),
        lr=5e-3,
        epochs=15,
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
                    num_authors=num_eval_authors,
                    author_offset=author_offset,
                    seed=42,
                ),
                name_for_wandb=f"tofu_modular_{cartridge_name}",
                generate_max_new_tokens=256,
                batch_size=min(32, num_eval_authors * 20),
                temperature=0.0,
            )
        ],
        distributed_backend="gloo",
        save_every_n_steps=256,
        save_after_training=True,
        wandb=WandBConfig(
            tags=["tofu", "modular", cartridge_name, f"n{num_eval_authors}", f"r{num_tokens}", f"targets_{TARGETS}"],
            notes=f"corpus_num_tokens={corpus_num_tokens}",
        ),
        output_dir=output_dir,
        name=FormatStringVariable(
            f"tofu_modular_{cartridge_name}_r{num_tokens}_lr{{lr}}"
        ),
    )


def compose_caches(cache_a_path: str, cache_b_path: str, device: str = "cuda"):
    """
    Compose two trained cartridges by concatenating their KV caches.

    Both caches must share the same AttnConfig (n_layers, n_heads, head_dim).
    The composed cache has (tokens_A + tokens_B) trainable tokens, all with
    seq_id = CARTRIDGE_SEQ_ID (-1), so every query token attends to the
    entire composed cache.

    Args:
        cache_a_path: Path to Cartridge A checkpoint (cache_last.pt).
        cache_b_path: Path to Cartridge B checkpoint (cache_last.pt).
        device: Device to load onto.

    Returns:
        A single TrainableCache with concatenated keys and values.
    """
    from cartridges.cache import TrainableCache

    cache_a = TrainableCache.from_pretrained(cache_a_path, device=device)
    cache_b = TrainableCache.from_pretrained(cache_b_path, device=device)

    assert cache_a.config.n_layers == cache_b.config.n_layers
    assert cache_a.config.n_heads == cache_b.config.n_heads
    assert cache_a.config.head_dim == cache_b.config.head_dim

    n_layers = cache_a.config.n_layers

    # Concatenate frozen keys/values (if any)
    has_frozen_a = cache_a._num_frozen_tokens > 0
    has_frozen_b = cache_b._num_frozen_tokens > 0

    # Build per-layer init_keys and init_values: [frozen_a, frozen_b, trainable_a, trainable_b]
    init_keys = []
    init_values = []
    num_frozen = 0

    for layer_idx in range(n_layers):
        parts_k, parts_v = [], []

        # Frozen parts
        if has_frozen_a:
            parts_k.append(cache_a.frozen_keys[layer_idx].data)
            parts_v.append(cache_a.frozen_values[layer_idx].data)
        if has_frozen_b:
            parts_k.append(cache_b.frozen_keys[layer_idx].data)
            parts_v.append(cache_b.frozen_values[layer_idx].data)

        # Trainable parts
        parts_k.append(cache_a.trainable_keys[layer_idx].data)
        parts_v.append(cache_a.trainable_values[layer_idx].data)
        parts_k.append(cache_b.trainable_keys[layer_idx].data)
        parts_v.append(cache_b.trainable_values[layer_idx].data)

        init_keys.append(torch.cat(parts_k, dim=2).contiguous())
        init_values.append(torch.cat(parts_v, dim=2).contiguous())

    num_frozen = cache_a._num_frozen_tokens + cache_b._num_frozen_tokens

    composed = TrainableCache(
        config=cache_a.config,
        init_keys=init_keys,
        init_values=init_values,
        num_frozen_tokens=num_frozen,
    ).to(device)

    total_a = cache_a._num_frozen_tokens + cache_a._num_trainable_tokens
    total_b = cache_b._num_frozen_tokens + cache_b._num_trainable_tokens
    print(f"  Composed cache: {total_a} (A) + {total_b} (B) = {total_a + total_b} tokens")

    del cache_a, cache_b
    return composed


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
    """
    Evaluate a single cache (loaded or composed) by running generation
    over the specified authors' questions.

    Args:
        cache: A TrainableCache (already on device).
        model: The causal LM (already on device).
        tokenizer: Tokenizer matching the model.
        num_authors: Number of authors to evaluate.
        author_offset: Skip this many authors (for evaluating subsets).
        label: Label for progress bar / logging.
        device: Device string.
        max_new_tokens: Max tokens to generate per question.
        temperature: Sampling temperature (0 = greedy).
        batch_size: Batch size for generation.

    Returns:
        DataFrame with per-question results including rouge_l scores.
    """
    from cartridges.generation import flex_generate
    from cartridges.data.tofu.evals import TOFUQAGenerateDataset

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


def evaluate_single_cartridge(
    cache_path: str,
    num_authors: int,
    author_offset: int = 0,
    label: str = "cartridge",
    model=None,
    tokenizer=None,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Load a single cache from disk and evaluate it."""
    from cartridges.cache import TrainableCache

    cache = TrainableCache.from_pretrained(cache_path, device=device)
    return evaluate_cache(
        cache=cache,
        model=model,
        tokenizer=tokenizer,
        num_authors=num_authors,
        author_offset=author_offset,
        label=label,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
    )


def evaluate_composed_cartridges(
    cache_a_path: str,
    cache_b_path: str,
    num_authors: int,
    model=None,
    tokenizer=None,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Compose two caches by concatenation and evaluate all authors."""
    composed_cache = compose_caches(cache_a_path, cache_b_path, device=device)
    return evaluate_cache(
        cache=composed_cache,
        model=model,
        tokenizer=tokenizer,
        num_authors=num_authors,
        label="composed",
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
    )


def print_summary(summary: Dict[str, pd.DataFrame]):
    """
    Print a nicely formatted comparison table of all cartridge evaluations.

    Args:
        summary: Mapping of cartridge name -> results DataFrame.
    """
    # Collect all score columns across all DataFrames
    all_score_cols = set()
    for df in summary.values():
        all_score_cols.update(
            col for col in df.columns if "rouge" in col.lower() or col == "score"
        )
    score_cols = sorted(all_score_cols)
    if not score_cols:
        print("\n  No score columns found in results.")
        return

    # Build header
    name_width = max(len(name) for name in summary) + 2
    col_width = 12
    header = f"{'Cartridge':<{name_width}}" + "".join(
        f"{col:>{col_width}}" for col in score_cols
    ) + f"{'n_questions':>{col_width}}"
    separator = "-" * len(header)

    print("\n" + separator)
    print("  MODULAR COMPARISON SUMMARY")
    print(f"  {NUM_AUTHORS} authors | R={NUM_TOKENS} tokens | targets={TARGETS} | model={MODEL}")
    print(separator)
    print(header)
    print(separator)

    for name, df in summary.items():
        row = f"{name:<{name_width}}"
        for col in score_cols:
            if col in df.columns:
                row += f"{df[col].mean():>{col_width}.4f}"
            else:
                row += f"{'N/A':>{col_width}}"
        row += f"{len(df):>{col_width}}"
        print(row)

    print(separator)


def main():
    all_authors = load_tofu_authors(num_authors=NUM_AUTHORS, seed=42)
    half = NUM_AUTHORS // 2
    half_tokens = NUM_TOKENS // 2

    print(f"Modular comparison: {NUM_AUTHORS} authors, R={NUM_TOKENS}, targets={TARGETS}")
    print(f"  Monolithic: 1 cartridge, all {NUM_AUTHORS} authors, R={NUM_TOKENS}")
    print(f"  Modular A:  authors 0-{half - 1}, R={half_tokens}")
    print(f"  Modular B:  authors {half}-{NUM_AUTHORS - 1}, R={half_tokens}")

    # --- Load model once for rescoring (if needed) ---
    rescore_model, rescore_tokenizer = None, None
    if TARGETS == "logits":
        print("\nLoading base model for rescoring...")
        rescore_model = model_config.instantiate().to("cuda").to(torch.bfloat16)
        rescore_tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name_or_path)

    # --- Prepare all data up front (rescore once, train multiple times) ---
    print("\n=== Preparing data ===")
    mono_path, mono_corpus, mono_ntok = prepare_data(
        all_authors, f"monolithic_n{NUM_AUTHORS}",
        rescore_model, rescore_tokenizer,
    )
    a_path, a_corpus, a_ntok = prepare_data(
        all_authors[:half], f"A_n{half}",
        rescore_model, rescore_tokenizer,
    )
    b_path, b_corpus, b_ntok = prepare_data(
        all_authors[half:], f"B_n{NUM_AUTHORS - half}",
        rescore_model, rescore_tokenizer,
    )

    # Free rescoring model before training
    if rescore_model is not None:
        del rescore_model
        torch.cuda.empty_cache()

    # --- Train all cartridges ---
    from cartridges.train import train

    # Build configs first (need run_dir for cache paths after training)
    mono_config = make_train_config(
        mono_path, mono_corpus, mono_ntok,
        cartridge_name=f"monolithic_n{NUM_AUTHORS}",
        num_tokens=NUM_TOKENS,
        num_eval_authors=NUM_AUTHORS,
        author_offset=0,
        num_conversations=NUM_AUTHORS * 20,
    )
    a_config = make_train_config(
        a_path, a_corpus, a_ntok,
        cartridge_name=f"A_n{half}",
        num_tokens=half_tokens,
        num_eval_authors=half,
        author_offset=0,
        num_conversations=half * 20,
    )
    b_config = make_train_config(
        b_path, b_corpus, b_ntok,
        cartridge_name=f"B_n{NUM_AUTHORS - half}",
        num_tokens=half_tokens,
        num_eval_authors=NUM_AUTHORS - half,
        author_offset=half,
        num_conversations=(NUM_AUTHORS - half) * 20,
    )

    print("\n=== Training Monolithic Cartridge ===")
    pydrantic.main(mono_config)

    print(f"\n=== Training Cartridge A (authors 0-{half - 1}) ===")
    pydrantic.main(a_config)

    print(f"\n=== Training Cartridge B (authors {half}-{NUM_AUTHORS - 1}) ===")
    pydrantic.main(b_config)

    # --- Step 4: Unified post-training evaluation ---
    cache_mono_path = os.path.join(mono_config.run_dir, "cache_last.pt")
    cache_a_path = os.path.join(a_config.run_dir, "cache_last.pt")
    cache_b_path = os.path.join(b_config.run_dir, "cache_last.pt")

    missing = []
    for name, path in [("Monolithic", cache_mono_path), ("A", cache_a_path), ("B", cache_b_path)]:
        if not os.path.exists(path):
            missing.append(f"  {name}: {path}")
    if missing:
        print(f"\nWARNING: Cache files not found. Skipping evaluation.")
        for m in missing:
            print(m)
        return

    print("\n=== Post-training evaluation (all cartridges on all authors) ===")

    # Load model + tokenizer once for all evaluations
    device = "cuda"
    model = model_config.instantiate().to(device).to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name_or_path)
    eval_batch_size = min(32, NUM_AUTHORS * 20)

    summary: Dict[str, pd.DataFrame] = {}

    # Monolithic: all authors, budget R
    print(f"\n--- Monolithic (all {NUM_AUTHORS} authors, R={NUM_TOKENS}) ---")
    summary[f"Monolithic (R={NUM_TOKENS})"] = evaluate_single_cartridge(
        cache_path=cache_mono_path,
        num_authors=NUM_AUTHORS,
        label="monolithic",
        model=model, tokenizer=tokenizer,
        device=device, batch_size=eval_batch_size,
    )

    # Cartridge A: first half of authors, budget R/2
    print(f"\n--- Cartridge A (authors 0-{half-1}, R={half_tokens}) ---")
    summary[f"Cartridge A (R={half_tokens})"] = evaluate_single_cartridge(
        cache_path=cache_a_path,
        num_authors=half,
        author_offset=0,
        label="cart-A",
        model=model, tokenizer=tokenizer,
        device=device, batch_size=eval_batch_size,
    )

    # Cartridge B: second half of authors, budget R/2
    print(f"\n--- Cartridge B (authors {half}-{NUM_AUTHORS-1}, R={half_tokens}) ---")
    summary[f"Cartridge B (R={half_tokens})"] = evaluate_single_cartridge(
        cache_path=cache_b_path,
        num_authors=NUM_AUTHORS - half,
        author_offset=half,
        label="cart-B",
        model=model, tokenizer=tokenizer,
        device=device, batch_size=eval_batch_size,
    )

    # Composed: concatenate A+B, evaluate all authors
    print(f"\n--- Composed A+B (all {NUM_AUTHORS} authors, R={half_tokens}+{half_tokens}) ---")
    summary[f"Composed A+B (R={half_tokens}+{half_tokens})"] = evaluate_composed_cartridges(
        cache_a_path=cache_a_path,
        cache_b_path=cache_b_path,
        num_authors=NUM_AUTHORS,
        model=model, tokenizer=tokenizer,
        device=device, batch_size=eval_batch_size,
    )

    # Clean up
    del model
    torch.cuda.empty_cache()

    # Save composed results
    composed_df = summary[f"Composed A+B (R={half_tokens}+{half_tokens})"]
    composed_path = os.path.join(output_dir, f"tofu_composed_n{NUM_AUTHORS}_r{NUM_TOKENS}.parquet")
    composed_df.to_parquet(composed_path)

    # Log to wandb
    try:
        import wandb
        from cartridges.utils.wandb import prepare_wandb

        wandb_config = WandBConfig(
            tags=["tofu", "modular", "composed", f"n{NUM_AUTHORS}", f"r{NUM_TOKENS}", f"targets_{TARGETS}"],
            notes=f"Composed: concatenate A (R={half_tokens}) + B (R={half_tokens}), total R={NUM_TOKENS}",
            name=f"tofu_composed_n{NUM_AUTHORS}_r{NUM_TOKENS}",
        )
        prepare_wandb(wandb_config, {
            "num_authors": NUM_AUTHORS,
            "num_tokens": NUM_TOKENS,
            "targets": TARGETS,
            "model": MODEL,
            "half": half,
            "half_tokens": half_tokens,
        })

        score_cols = [col for col in composed_df.columns if "rouge" in col.lower() or col == "score"]
        log_dict = {f"composed/{col}": composed_df[col].mean() for col in score_cols}
        log_dict["composed/table"] = composed_df
        wandb.log(log_dict)
        wandb.finish()
    except Exception as e:
        print(f"  Warning: Failed to log to wandb: {e}")

    # --- Print final summary ---
    print_summary(summary)


if __name__ == "__main__":
    main()
