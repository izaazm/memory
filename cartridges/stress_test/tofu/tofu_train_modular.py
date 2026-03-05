"""
Modular cartridge comparison for TOFU capacity stress test.

Splits N authors into 2 groups and trains separate cartridges (A and B).
Evaluates with manual routing: each question is directed to the correct
cartridge based on author assignment.

Compares:
- Monolithic: 1 cartridge with all N authors at budget R
- Modular: 2 cartridges with N/2 authors each at budget R/2

Usage:
    NUM_AUTHORS=10 NUM_TOKENS=64 python stress_test/tofu/tofu_train_modular.py

TODO: This is a scaffold — implement cartridge composition/routing.
"""
import os
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
MODEL = os.environ.get("MODEL", "llama")
output_dir = os.environ.get("CARTRIDGES_OUTPUT_DIR", ".")


def train_cartridge(
    authors,
    cartridge_name: str,
    num_tokens: int,
    eval_num_authors: int,
):
    """Train a single cartridge on a subset of authors."""
    conversations = authors_to_conversations(authors)
    train_path = os.path.join(output_dir, f"tofu_modular_{cartridge_name}.parquet")
    save_conversations_to_parquet(conversations, train_path)
    corpus_path = save_corpus_to_tempfile(authors)

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

    num_eval_authors = len(authors)
    config = TrainConfig(
        model=model_config,
        kv_cache_initializer=KVFromText.Config(
            text_source=corpus_path,
            max_tokens=num_tokens,
        ),
        lr=2e-2,
        epochs=2,
        global_batch_size=min(32, len(conversations)),
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
                    num_authors=num_eval_authors,
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
            tags=["tofu", "modular", cartridge_name, f"r{num_tokens}"]
        ),
        output_dir=output_dir,
        name=FormatStringVariable(
            f"tofu_modular_{cartridge_name}_r{num_tokens}"
        ),
    )

    from cartridges.train import train
    train(config)


def main():
    """
    Modular comparison:
    1. Train monolithic cartridge: all N authors, budget R
    2. Train cartridge A: authors 0..N/2-1, budget R/2
    3. Train cartridge B: authors N/2..N-1, budget R/2
    4. Compare: monolithic vs composed (A + B with routing)
    
    Note: Step 4 (composition) requires cartridge serving infrastructure.
    For now, we evaluate each cartridge independently and compare.
    """
    all_authors = load_tofu_authors(num_authors=NUM_AUTHORS, seed=42)
    half = NUM_AUTHORS // 2
    half_tokens = NUM_TOKENS // 2

    print(f"Modular comparison: {NUM_AUTHORS} authors, R={NUM_TOKENS}")
    print(f"  Monolithic: 1 cartridge, all {NUM_AUTHORS} authors, R={NUM_TOKENS}")
    print(f"  Modular A:  authors 0-{half - 1}, R={half_tokens}")
    print(f"  Modular B:  authors {half}-{NUM_AUTHORS - 1}, R={half_tokens}")

    # 1. Monolithic
    print("\n=== Training Monolithic Cartridge ===")
    train_cartridge(
        all_authors,
        cartridge_name=f"monolithic_n{NUM_AUTHORS}",
        num_tokens=NUM_TOKENS,
        eval_num_authors=NUM_AUTHORS,
    )

    # 2. Cartridge A (first half)
    print(f"\n=== Training Cartridge A (authors 0-{half - 1}) ===")
    train_cartridge(
        all_authors[:half],
        cartridge_name=f"A_n{half}",
        num_tokens=half_tokens,
        eval_num_authors=half,
    )

    # 3. Cartridge B (second half)
    print(f"\n=== Training Cartridge B (authors {half}-{NUM_AUTHORS - 1}) ===")
    train_cartridge(
        all_authors[half:],
        cartridge_name=f"B_n{NUM_AUTHORS - half}",
        num_tokens=half_tokens,
        eval_num_authors=NUM_AUTHORS - half,
    )

    print("\n=== All modular training complete ===")
    print("Compare wandb results for monolithic vs modular (A + B).")
    print("TODO: Implement cartridge composition for joint evaluation.")


if __name__ == "__main__":
    main()
