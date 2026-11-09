"""Prepare and fine-tune a CPU-compatible Hugging Face URL classifier."""

import argparse
import csv
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "urls_processed.csv"


def format_example(row: dict[str, str]) -> str:
    page_fields = (
        "has_password_field",
        "has_external_form",
        "iframes_count",
        "redirect_count",
        "title_length",
        "is_https",
    )
    indicators = " ".join(
        f"{field} {row[field]}"
        for field in page_fields
        if row.get(field) not in (None, "")
    )
    return f"URL {row['url']} source {row.get('source', 'unknown')} {indicators}".strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_hf_dataset(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = ROOT / "data" / "processed" / "hf_training.jsonl",
    metadata_path: Path = ROOT / "data" / "metadata" / "hf_training_meta.json",
    max_per_class: int = 2000,
    seed: int = 42,
) -> dict:
    if max_per_class < 2:
        raise ValueError("max_per_class must be at least 2")
    if not input_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {input_path}")

    rng = random.Random(seed)
    samples: dict[int, list[dict[str, str]]] = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                label = int(row["label"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("Processed dataset must have a binary label column") from None
            if label not in samples:
                raise ValueError(f"Unsupported processed label: {label}")
            seen[label] += 1
            if len(samples[label]) < max_per_class:
                samples[label].append(row)
            else:
                index = rng.randrange(seen[label])
                if index < max_per_class:
                    samples[label][index] = row
    if any(len(rows) < 2 for rows in samples.values()):
        raise ValueError("Hugging Face training requires at least two rows per class")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for label in sorted(samples):
            for row in samples[label]:
                stream.write(json.dumps({
                    "text": format_example(row),
                    "label": label,
                }) + "\n")
    metadata = {
        "schema_version": 1,
        "source_file": input_path.name,
        "source_sha256": _sha256(input_path),
        "sample_count": sum(map(len, samples.values())),
        "label_counts": {str(label): len(rows) for label, rows in samples.items()},
        "max_per_class": max_per_class,
        "random_seed": seed,
        "text_fields": ["url", "source", "available_page_features"],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def train_hf_model(
    input_path: Path,
    output_dir: Path,
    metrics_path: Path,
    *,
    model_name: str = "distilbert-base-uncased",
    seed: int = 42,
    epochs: int = 1,
    batch_size: int = 8,
    source_dataset_sha256: str | None = None,
) -> dict:
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError(
            "The optional Hugging Face pipeline needs CPU PyTorch and packages from "
            "requirements-hf.txt. See README.md for CPU-only installation."
        ) from error

    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [record["text"] for record in records]
    labels = [int(record["label"]) for record in records]
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "legitimate", 1: "phishing"},
        label2id={"legitimate": 0, "phishing": 1},
    )

    class URLDataset(torch.utils.data.Dataset):
        def __init__(self, examples, target_labels):
            self.encodings = tokenizer(
                examples,
                truncation=True,
                max_length=128,
                padding=False,
            )
            self.labels = target_labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {
                key: torch.tensor(value[index])
                for key, value in self.encodings.items()
            }
            item["labels"] = torch.tensor(self.labels[index])
            return item

    train_dataset = URLDataset(train_texts, train_labels)
    eval_dataset = URLDataset(eval_texts, eval_labels)

    def compute_metrics(prediction):
        import numpy as np

        predicted = np.argmax(prediction.predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            prediction.label_ids,
            predicted,
            average="binary",
            zero_division=0,
        )
        return {
            "accuracy": accuracy_score(prediction.label_ids, predicted),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=100,
        report_to=[],
        use_cpu=True,
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    evaluation = trainer.evaluate()
    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)

    dataset_metadata = {
        "model_name": model_name,
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "source_dataset_sha256": source_dataset_sha256 or _sha256(input_path),
        "training_sample_sha256": _sha256(input_path),
        "train_rows": len(train_texts),
        "validation_rows": len(eval_texts),
        "random_seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_metrics": {
            key.removeprefix("eval_"): float(value)
            for key, value in evaluation.items()
            if key.startswith("eval_") and isinstance(value, (int, float))
        },
        "tags": {
            "framework": "huggingface-transformers",
            "runtime": "cpu",
            "validation": "stratified_holdout",
        },
    }
    (output_dir / "model_manifest.json").write_text(
        json.dumps(dataset_metadata, indent=2),
        encoding="utf-8",
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(dataset_metadata, indent=2), encoding="utf-8")

    try:
        import mlflow
    except ImportError:
        return dataset_metadata
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or (
        f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("phishing-url-analysis-huggingface")
    with mlflow.start_run(run_name="distilbert-url-classifier"):
        mlflow.log_params({
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "random_seed": seed,
            "dataset_sha256": dataset_metadata["source_dataset_sha256"],
        })
        mlflow.log_metrics({
            key: value
            for key, value in dataset_metadata["validation_metrics"].items()
        })
        mlflow.set_tags(dataset_metadata["tags"])
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        mlflow.log_artifact(str(output_dir / "model_manifest.json"))
    return dataset_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--metadata", type=Path, required=True)
    prepare.add_argument("--max-per-class", type=int, default=2000)
    train = subparsers.add_parser("train")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--metrics", type=Path, required=True)
    train.add_argument("--model-name", default="distilbert-base-uncased")
    train.add_argument("--epochs", type=int, default=1)
    run = subparsers.add_parser("run")
    run.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    run.add_argument("--dataset-output", type=Path, default=ROOT / "data" / "processed" / "hf_training.jsonl")
    run.add_argument("--metadata", type=Path, default=ROOT / "data" / "metadata" / "hf_training_meta.json")
    run.add_argument("--output", type=Path, default=ROOT / "models" / "huggingface")
    run.add_argument("--metrics", type=Path, default=ROOT / "reports" / "hf_evaluation.json")
    run.add_argument("--model-name", default="distilbert-base-uncased")
    run.add_argument("--max-per-class", type=int, default=2000)
    run.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare_hf_dataset(
            args.input,
            args.output,
            args.metadata,
            args.max_per_class,
        )
    elif args.command == "train":
        result = train_hf_model(
            args.input,
            args.output,
            args.metrics,
            model_name=args.model_name,
            epochs=args.epochs,
        )
    else:
        preparation = prepare_hf_dataset(
            args.input,
            args.dataset_output,
            args.metadata,
            args.max_per_class,
        )
        result = train_hf_model(
            args.dataset_output,
            args.output,
            args.metrics,
            model_name=args.model_name,
            epochs=args.epochs,
            source_dataset_sha256=preparation["source_sha256"],
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
