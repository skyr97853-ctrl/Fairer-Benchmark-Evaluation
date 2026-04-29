import argparse
import json

from datasets import load_dataset


CUSTOM_INSTRUCTION = (
    "Read the following science question and select the most appropriate answer "
    "from the choices provided. Output only the label corresponding to the correct "
    "option (e.g., 'A', 'B', 'C', or 'D')."
)


def load_openbookqa(dataset_name, split):
    config = "main"
    return load_dataset(dataset_name, config, split=split)


def format_input(row):
    question = row.get("question_stem", "").strip()
    choices = row.get("choices", {})
    option_lines = [
        f"{label}: {text.strip()}"
        for label, text in zip(choices.get("label", []), choices.get("text", []))
    ]
    return f"Question:\n{question}\n\nOptions:\n{chr(10).join(option_lines)}"


def build_normal_entry(row):
    return {
        "instruction": CUSTOM_INSTRUCTION,
        "input": format_input(row),
        "output": row.get("answerKey", "").strip(),
    }


def build_reverse_entries(row):
    input_text = format_input(row)
    return [
        {
            "instruction": CUSTOM_INSTRUCTION,
            "input": input_text,
            "output": label,
        }
        for label in row.get("choices", {}).get("label", [])
    ]


def build_datasets(dataset):
    normal_data = []
    reverse_data = []

    for row in dataset:
        try:
            normal_data.append(build_normal_entry(row))
            reverse_data.extend(build_reverse_entries(row))
        except Exception:
            continue

    return normal_data, reverse_data


def save_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def process_openbookqa(
    dataset_name,
    split="test",
    normal_output_file="data_normal_openbookqa.json",
    reverse_output_file="data_reverse_openbookqa.json",
):
    try:
        dataset = load_openbookqa(dataset_name, split)
    except Exception:
        return

    normal_data, reverse_data = build_datasets(dataset)
    save_json(normal_data, normal_output_file)
    save_json(reverse_data, reverse_output_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="allenai/openbookqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--normal-output", default="data_normal_openbookqa.json")
    parser.add_argument("--reverse-output", default="data_reverse_openbookqa.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_openbookqa(
        dataset_name=args.dataset_name,
        split=args.split,
        normal_output_file=args.normal_output,
        reverse_output_file=args.reverse_output,
    )
