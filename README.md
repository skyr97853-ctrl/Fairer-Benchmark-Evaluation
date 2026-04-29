# Fairer Benchmark Evaluation

This project provides helper scripts for preparing benchmark datasets, running LoRA fine-tuning with LLaMA-Factory, and evaluating trained adapters.

## 1. Install LLaMA-Factory

Clone and install LLaMA-Factory first:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt
cd ..
```

The scripts in this project expect the LLaMA-Factory repository to exist at:

```text
./LLaMA-Factory
```

## 2. Prepare datasets

Edit the configuration variables at the top of `prepare_llamafactory_data.py`:

```python
DATASET_NAME = ""
SPLIT = ""
NORMAL_FILE = ""
REVERSE_FILE = ""
```

Example:

```python
DATASET_NAME = "allenai/openbookqa"
SPLIT = "test"
NORMAL_FILE = "data_normal_openbookqa.json"
REVERSE_FILE = "data_reverse_openbookqa.json"
```

Then run:

```bash
python prepare_llamafactory_data.py
```

This script runs `pre_datasets.py`, writes the generated dataset files to:

```text
./LLaMA-Factory/data
```

It also updates:

```text
./LLaMA-Factory/data/dataset_info.json
```

The dataset keys are generated from the output file names by removing `.json`. For example:

```text
data_normal_openbookqa.json  ->  data_normal_openbookqa
data_reverse_openbookqa.json ->  data_reverse_openbookqa
```

## 3. Train adapters

Use `run_train.sh` to run training. Before running it, fill in the variables at the top of the file:

```bash
DATASET=""
MODEL_PATH=""
ADAPTER_PATH=""
OUTPUT_DIR=""
```

`DATASET` should match a key in `LLaMA-Factory/data/dataset_info.json`.

Then run:

```bash
bash run_train.sh
```

Use this script for the three-stage training workflow by updating `DATASET`, `ADAPTER_PATH`, and `OUTPUT_DIR` for each stage.

## 4. Evaluate adapters

Use `eval.sh` for evaluation. Before running it, fill in:

```bash
DATASET=""
MODEL_PATH=""
ADAPTER_PATH=""
OUTPUT_DIR=""
```

`DATASET` should match the evaluation dataset key in `LLaMA-Factory/data/dataset_info.json`, and `ADAPTER_PATH` should point to the adapter or checkpoint to evaluate.

Then run:

```bash
bash eval.sh
```

Evaluation outputs are written to `OUTPUT_DIR`.
