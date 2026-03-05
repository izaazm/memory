"""
Continual learning variant of TOFU capacity stress test.

Instead of training on all N authors at once, incrementally adds
2 authors at a time to the same cartridge checkpoint.

Usage:
    NUM_AUTHORS=10 NUM_TOKENS=64 python stress_test/tofu/tofu_train_continual.py

TODO: This is a scaffold — implement with checkpoint loading/resuming.
"""
import os
import tempfile
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization import KVFromText
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.datasets import TrainDataset, DataSource
from cartridges.data.tofu.evals import TOFUQAGenerateDataset
from cartridges.data.tofu.utils import (
    load_tofu_authors,
    authors_to_conversations,
    save_corpus_to_tempfile,
    save_conversations_to_parquet,
)
from cartridges.utils.wandb import WandBConfig


NUM_AUTHORS = int(os.environ.get("NUM_AUTHORS", "10"))
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "64"))
AUTHORS_PER_STEP = int(os.environ.get("AUTHORS_PER_STEP", "2"))
MODEL = os.environ.get("MODEL", "llama")
output_dir = os.environ.get("CARTRIDGES_OUTPUT_DIR", ".")


def main():
    """
    Continual learning loop:
    1. Load all N authors
    2. For each step t (2 authors at a time):
       a. Create training data with authors [0..2t]
       b. If t > 0, load cartridge from previous checkpoint
       c. Fine-tune cartridge
       d. Evaluate on ALL authors seen so far
    """
    all_authors = load_tofu_authors(num_authors=NUM_AUTHORS, seed=42)
    num_steps = (NUM_AUTHORS + AUTHORS_PER_STEP - 1) // AUTHORS_PER_STEP

    print(f"Continual learning: {NUM_AUTHORS} authors in {num_steps} steps "
          f"({AUTHORS_PER_STEP} authors/step)")

    for step in range(num_steps):
        start_idx = step * AUTHORS_PER_STEP
        end_idx = min(start_idx + AUTHORS_PER_STEP, NUM_AUTHORS)
        cumulative_end = end_idx

        # Authors for this step's training data (new authors only)
        step_authors = all_authors[start_idx:end_idx]
        # All authors seen so far (for evaluation)
        all_seen = all_authors[:cumulative_end]

        print(f"\n--- Step {step + 1}/{num_steps}: "
              f"Training on authors {start_idx}-{end_idx - 1}, "
              f"Eval on authors 0-{cumulative_end - 1} ---")

        # Prepare training data (new authors only)
        step_conversations = authors_to_conversations(step_authors)
        train_path = os.path.join(
            output_dir,
            f"tofu_continual_step{step}_n{cumulative_end}.parquet"
        )
        save_conversations_to_parquet(step_conversations, train_path)

        # Corpus for KV init (all seen authors)
        corpus_path = save_corpus_to_tempfile(all_seen)

        # TODO: For step > 0, load cartridge from previous checkpoint
        # instead of re-initializing from text.
        # Use KVCacheFromPretrained or similar.

        if MODEL == "llama":
            from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
            from cartridges.models import HFModelConfig
            model_config = HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
                model_cls=FlexLlamaForCausalLM,
            )
        elif MODEL == "qwen":
            from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
            from cartridges.models import HFModelConfig
            model_config = HFModelConfig(
                pretrained_model_name_or_path="Qwen/Qwen3-4b",
                model_cls=FlexQwen3ForCausalLM,
            )

        config = TrainConfig(
            model=model_config,
            kv_cache_initializer=KVFromText.Config(
                text_source=corpus_path,
                max_tokens=NUM_TOKENS,
            ),
            lr=2e-2,
            epochs=2,
            global_batch_size=min(32, len(step_conversations)),
            dataset=TrainDataset.Config(
                data_sources=[DataSource(path=train_path, type="local")],
                targets="tokens",
                top_k_logits=20,
                packed_seq_length=2048,
                packing_mode="pad",
            ),
            generate_eval_every_n_steps=64,
            generate_evals=[
                GenerationEvalConfig(
                    dataset=TOFUQAGenerateDataset.Config(
                        num_authors=cumulative_end,
                        seed=42,
                    ),
                    name_for_wandb=f"tofu_continual_step{step}_n{cumulative_end}",
                    generate_max_new_tokens=256,
                    batch_size=min(32, cumulative_end * 20),
                    temperature=0.0,
                )
            ],
            distributed_backend="gloo",
            save_every_n_steps=256,
            save_after_training=True,
            wandb=WandBConfig(
                tags=["tofu", "continual", f"step{step}", f"n{cumulative_end}", f"r{NUM_TOKENS}"]
            ),
            output_dir=output_dir,
            name=FormatStringVariable(
                f"tofu_continual_step{step}_n{cumulative_end}_r{NUM_TOKENS}"
            ),
        )

        # Run training for this step
        from cartridges.train import train
        train(config)

        print(f"Step {step + 1} complete.")


if __name__ == "__main__":
    main()
