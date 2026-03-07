"""
Training script for TOFU capacity stress test.

Trains a single cartridge for a given (N_authors, R_tokens) configuration.
No synthesis step needed — TOFU QA pairs are used directly.

Usage:
    python stress_test/tofu/tofu_train.py
    python stress_test/tofu/tofu_train.py num_authors=5 kv_cache_initializer.max_tokens=64

Environment variables:
    CARTRIDGES_OUTPUT_DIR: Output directory for artifacts
    NUM_AUTHORS: Number of TOFU authors to train on (default: 2)
    NUM_TOKENS: Cartridge size in tokens (default: 64)
    MODEL: "llama" or "qwen" (default: "llama")
"""
import os
import sys
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


# --- Configuration from environment ---
os.environ["CARTRIDGES_WANDB_PROJECT"] = "your-wandb-project"
os.environ["CARTRIDGES_WANDB_ENTITY"] = "your-wandb-username-or-team"

NUM_AUTHORS = int(os.environ.get("NUM_AUTHORS", "2"))
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "64"))
MODEL = os.environ.get("MODEL", "llama")

# --- Model selection ---
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
else:
    raise ValueError(f"Invalid model: {MODEL}. Use 'llama' or 'qwen'.")


# --- Prepare TOFU data ---
# Load authors, convert to conversations, and save to parquet
authors = load_tofu_authors(num_authors=NUM_AUTHORS, seed=42)
conversations = authors_to_conversations(authors)

# Save training data as parquet
output_dir = os.environ.get("CARTRIDGES_OUTPUT_DIR", ".")
train_data_path = os.path.join(output_dir, f"tofu_train_n{NUM_AUTHORS}.parquet")
save_conversations_to_parquet(conversations, train_data_path)

# Save corpus text for KV cache initialization
corpus_path = save_corpus_to_tempfile(authors)


# --- Training config ---
config = TrainConfig(
    model=model_config,
    kv_cache_initializer=KVFromText.Config(
        text_source=corpus_path,
        max_tokens=NUM_TOKENS,
    ),

    lr=5e-4,
    epochs=3,
    global_batch_size=min(32, len(conversations)),

    dataset=TrainDataset.Config(
        data_sources=[
            DataSource(path=train_data_path, type="local"),
        ],
        # Use token-level targets since we don't have synthesized logprobs
        targets="tokens",
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="pad",
    ),

    # Evaluate generation every N steps
    generate_eval_every_n_steps=64,
    generate_evals=[
        GenerationEvalConfig(
            dataset=TOFUQAGenerateDataset.Config(
                num_authors=NUM_AUTHORS,
                seed=42,
            ),
            name_for_wandb=f"tofu_n{NUM_AUTHORS}_r{NUM_TOKENS}",
            generate_max_new_tokens=256,
            batch_size=min(32, NUM_AUTHORS * 20),
            temperature=0.0,
        )
    ],

    distributed_backend="gloo",

    print_loss_every_n_steps=10,

    save_every_n_steps=256,
    save_after_training=True,

    wandb=WandBConfig(tags=["tofu", "capacity", f"n{NUM_AUTHORS}", f"r{NUM_TOKENS}"]),
    output_dir=output_dir,
    name=FormatStringVariable(
        f"tofu_capacity_n{NUM_AUTHORS}_r{NUM_TOKENS}_lr{{lr}}"
    ),
)


if __name__ == "__main__":
    pydrantic.main(config)
