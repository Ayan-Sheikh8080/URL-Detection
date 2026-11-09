import json

from src.training.evaluate import generate_detailed_report


def test_detailed_evaluation_reads_current_training_report_schema(tmp_path):
    source = tmp_path / "evaluation.json"
    output = tmp_path / "detailed.json"
    source.write_text(json.dumps({
        "headline": "stratified holdout",
        "experiment_tracking": "mlflow",
        "dataset_info": {"url_training_rows": 10},
        "configurations": {
            "url_only": {
                "sample_count": 10,
                "best_model": "random_forest",
                "best_holdout_f1": 0.8,
                "models": {
                    "logistic_regression": {"f1": 0.7, "precision": 0.8, "recall": 0.6},
                    "random_forest": {"f1": 0.8, "precision": 0.9, "recall": 0.7},
                },
            },
            "combined": {
                "sample_count": 10,
                "best_model": "random_forest",
                "best_holdout_f1": 0.9,
                "models": {
                    "random_forest": {"f1": 0.9, "precision": 0.95, "recall": 0.85},
                },
            },
        },
    }), encoding="utf-8")

    report = generate_detailed_report(source, output)

    assert report["rankings"]["url_only"][0]["model"] == "random_forest"
    assert round(report["improvement_comparison"]["absolute_change"], 4) == 0.1
    assert json.loads(output.read_text(encoding="utf-8"))["experiment_tracking"] == "mlflow"
