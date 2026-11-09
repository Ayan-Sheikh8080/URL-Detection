"""Summarize the stratified holdout report written by model training."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATH = ROOT / "reports" / "evaluation.json"
REPORT_PATH = ROOT / "reports" / "evaluation_detailed.json"


def load_evaluation(path: Path = EVALUATION_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def generate_detailed_report(
    evaluation_path: Path = EVALUATION_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    report = load_evaluation(evaluation_path)
    configurations = report.get("configurations", {})
    detailed: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_info": report.get("dataset_info", {}),
        "experiment_tracking": report.get("experiment_tracking", "unknown"),
        "headline_metric": report.get("headline", ""),
        "configuration_results": {},
        "rankings": {},
    }

    for config_name, config in configurations.items():
        models = [
            {
                "model": model_name,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "training_seconds": float(metrics.get("training_seconds", 0.0)),
                "test_rows": int(metrics.get("test_rows", 0)),
            }
            for model_name, metrics in config.get("models", {}).items()
        ]
        models.sort(key=lambda result: (-result["f1"], result["model"]))
        detailed["configuration_results"][config_name] = {
            "sample_count": config.get("sample_count", 0),
            "best_model": config.get("best_model"),
            "best_holdout_f1": config.get("best_holdout_f1"),
            "models": models,
        }
        detailed["rankings"][config_name] = models

    url_f1 = configurations.get("url_only", {}).get("best_holdout_f1")
    combined_f1 = configurations.get("combined", {}).get("best_holdout_f1")
    if url_f1 is not None and combined_f1 is not None:
        url_f1 = float(url_f1)
        combined_f1 = float(combined_f1)
        detailed["improvement_comparison"] = {
            "url_only_best_f1": url_f1,
            "combined_best_f1": combined_f1,
            "absolute_change": combined_f1 - url_f1,
            "improvement_percent": (
                (combined_f1 - url_f1) / url_f1 * 100 if url_f1 else 0.0
            ),
        }
    else:
        detailed["improvement_comparison"] = {
            "note": "URL-only and combined holdout results are both required"
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(detailed, indent=2), encoding="utf-8")
    return detailed


def main() -> int:
    report = generate_detailed_report()
    print(json.dumps({
        "output": str(REPORT_PATH),
        "configurations": list(report["configuration_results"]),
        "improvement_comparison": report["improvement_comparison"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
