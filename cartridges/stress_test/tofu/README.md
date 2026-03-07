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

### 3. Extended Experiments (Scaffolds)
For later stages of the research, two fully-configured scaffolds have been provided:
- [**tofu_train_continual.py**](file:///Users/izaazm/Documents/memory/cartridges/stress_test/tofu/tofu_train_continual.py): For continual learning validation vs full write-at-once.
- [**tofu_train_modular.py**](file:///Users/izaazm/Documents/memory/cartridges/stress_test/tofu/tofu_train_modular.py): Trains smaller distinct sub-cartridges (N/2 each) and a large monolithic cartridge for composition comparisons.

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
NUM_AUTHORS=20 NUM_TOKENS=16 MODEL=llama python stress_test/tofu/tofu_train.py
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
