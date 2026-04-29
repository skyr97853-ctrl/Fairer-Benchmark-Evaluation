#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


DATASET_NAME = ""
SPLIT = ""
NORMAL_FILE = ""
REVERSE_FILE = ""

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "LLaMA-Factory" / "data"
DATASET_INFO = DATA_DIR / "dataset_info.json"
PRE_DATASETS = SCRIPT_DIR / "pre_datasets.py"

DATASET_COLUMNS = {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
}


def run_pre_datasets(args, normal_output, reverse_output):
    command = [
        sys.executable,
        str(PRE_DATASETS),
        "--dataset-name",
        args.dataset_name,
        "--split",
        args.split,
        "--normal-output",
        str(normal_output),
        "--reverse-output",
        str(reverse_output),
    ]
    subprocess.run(command, check=True)


def ensure_outputs_exist(*paths):
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Generated dataset file not found: {', '.join(missing)}")


def load_dataset_info():
    with DATASET_INFO.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_dataset_info(dataset_info):
    with DATASET_INFO.open("w", encoding="utf-8") as file:
        json.dump(dataset_info, file, indent=2, ensure_ascii=False)
        file.write("\n")


def dataset_entry(file_name):
    return {
        "file_name": file_name,
        "columns": DATASET_COLUMNS,
    }


def dataset_key(file_name):
    return Path(file_name).stem


def update_dataset_info(normal_file_name, reverse_file_name):
    dataset_info = load_dataset_info()
    dataset_info[dataset_key(normal_file_name)] = dataset_entry(normal_file_name)
    dataset_info[dataset_key(reverse_file_name)] = dataset_entry(reverse_file_name)
    save_dataset_info(dataset_info)


def configured_value(arg_value, config_value):
    return arg_value if arg_value else config_value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name")
    parser.add_argument("--split")
    parser.add_argument("--normal-file")
    parser.add_argument("--reverse-file")
    args = parser.parse_args()
    args.dataset_name = configured_value(args.dataset_name, DATASET_NAME)
    args.split = configured_value(args.split, SPLIT)
    args.normal_file = configured_value(args.normal_file, NORMAL_FILE)
    args.reverse_file = configured_value(args.reverse_file, REVERSE_FILE)
    return args


def validate_args(args):
    required_args = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "normal_file": args.normal_file,
        "reverse_file": args.reverse_file,
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")


def main():
    args = parse_args()
    validate_args(args)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    normal_output = DATA_DIR / args.normal_file
    reverse_output = DATA_DIR / args.reverse_file

    run_pre_datasets(args, normal_output, reverse_output)
    ensure_outputs_exist(normal_output, reverse_output)
    update_dataset_info(normal_output.name, reverse_output.name)


if __name__ == "__main__":
    main()
