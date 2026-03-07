# Cartridge Capacity Stress Test: TOFU Experiment Walkthrough

## Implemented Components

### 1. TOFU Data Module (`cartridges/data/tofu/`)
- [**utils.py**](file:///Users/izaazm/Documents/memory/cartridges/cartridges/data/tofu/utils.py): Downloads the `locuslab/TOFU` dataset from HuggingFace, clusters the pairs into distinct authors (20 QA each), and converts them into `Conversation` objects for direct training. It also builds a flattened corpus text for `KVFromText` cartridge initialization.
- [**evals.py**](file:///Users/izaazm/Documents/memory/cartridges/cartridges/data/tofu/evals.py): Implements `TOFUQAGenerateDataset`. It fetches the exact same QA pairs used in training and scores predictions against ground truth using ROUGE-L.

### 2. Experiment Orchestration (`stress_test/tofu/`)
- [**tofu_train.py**](file:///Users/izaazm/Documents/memory/cartridges/stress_test/tofu/tofu_train.py): The core single-run training script. Set `NUM_AUTHORS` and `NUM_TOKENS` (R). It converts data to parquet formats dynamically, enforces `targets="tokens"` to side-step logprob dependencies, and evaluates using generation via `TOFUQAGenerateDataset`.
- [**tofu_sweep.py**](file:///Users/izaazm/Documents/memory/cartridges/stress_test/tofu/tofu_sweep.py): Sweep runner. It builds the $(N, R)$ matrix commands.
  - Defaults to $N \in \{1, 2, 5, 10, 20, 50\}$ and $R \in \{16, 32, 64, 128\}$, yielding 24 runs.
  - `python stress_test/tofu/tofu_sweep.py --dry-run` previews the sequence.
- [**tofu_analyze.py**](file:///Users/izaazm/Documents/memory/cartridges/stress_test/tofu/tofu_analyze.py): Post-training analysis script. It plots `Acc(N, R)` curves, derives $N^*(R)$, and outputs the storage efficiency (facts-per-byte) metrics.

### 3. Extended Experiments
- [**tofu_train_continual.py**](file:///Users/izaazm/Documents/memory/cartridges/stress_test/tofu/tofu_train_continual.py): For continual learning validation vs full write-at-once.
- [**tofu_train_modular.py**](file:///Users/izaazm/Documents/memory/cartridges/stress_test/tofu/tofu_train_modular.py): Modular composition experiment. Trains two sub-cartridges (A and B, each with N/2 authors at R/2 tokens) plus a monolithic cartridge (all N authors at R tokens). After training, concatenates A+B's KV caches into a single composed cache and evaluates all N authors' questions against it. This tests true composability: whether two independently trained cartridges, when concatenated without any routing, can match or exceed a monolithic cartridge of the same total size.
  - **Composition**: Both caches' key/value tensors are concatenated along the sequence dimension. All tokens retain `seq_id = -1` (CARTRIDGE_SEQ_ID), so every query token attends to the full composed cache.
  - **Evaluation**: The composed cache is evaluated on all N authors using ROUGE-L, giving a direct comparison against the monolithic baseline.
  - Supports `TARGETS=logits` (distillation) or `TARGETS=tokens` (SFT).

## How to Run the Experiments

These scripts require a CUDA GPU, so they should be executed on your training server. 

### 0. Environment Setup
setup from cartridges folder
```bash
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . 
```

### 1. Test a single run (Smoke Test)
Run a single training execution for 5 authors with a 64-token budget:
```bash
NUM_AUTHORS=5 NUM_TOKENS=16 MODEL=qwen python stress_test/tofu/tofu_train.py
```

### 2. Run the Full Sweep
Execute the full grid defined in the script (will output sequential progress):
```bash
python stress_test/tofu/tofu_sweep.py
```
*Note: You can pass `--n` and `--r` lists directly to override the grid sizing.*

### 3. Generate Analysis Plots
After the sweep finishes (or reading directly from your Weights & Biases entity):
```bash
python stress_test/tofu/tofu_analyze.py --wandb-entity WANDB_ENTITY --wandb-project WANDB_PROJECT
```

### 4. Run the Modular Composition Experiment
Trains a monolithic cartridge (N authors, R tokens) and two sub-cartridges (N/2 authors, R/2 tokens each), then concatenates the sub-cartridges and evaluates on all N authors:
```bash
NUM_AUTHORS=2 NUM_TOKENS=512 MODEL=llama python stress_test/tofu/tofu_train_modular.py
```
Use `TARGETS=tokens` for SFT (no rescoring, default) or `TARGETS=logits` for distillation

The script produces three W&B runs (monolithic, A, B) plus a composed evaluation run. Compare `generate_tofu_modular_monolithic_*/rouge_l_score` against `composed/rouge_l` to see if modular composition matches monolithic performance.
