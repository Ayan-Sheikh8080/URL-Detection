import json
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
EVALUATION_PATH = Path("reports/evaluation.json")
REPORT_PATH = Path("reports/evaluation_detailed.json")
MODELS_DIR = Path("models")


def load_evaluation():
    """Load the evaluation report generated during training."""
    if not EVALUATION_PATH.exists():
        print(f"Evaluation report not found at {EVALUATION_PATH}")
        return None
    
    with open(EVALUATION_PATH, 'r') as f:
        report = json.load(f)
    
    return report


def generate_detailed_report():
    """Generate a detailed evaluation report with comparisons."""
    report = load_evaluation()
    if report is None:
        return None
    
    results = report.get('results', {})
    improvement = report.get('improvement', {})
    
    detailed = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'dataset_info': report.get('dataset_info', {}),
        'headline_metric': report.get('headline', ''),
        'improvement_analysis': improvement,
        'configuration_results': {},
        'rankings': {}
    }
    
    # Organize results by configuration
    configs = {'url_only': [], 'content_only': [], 'combined': []}
    
    for config_name, result in results.items():
        config_key = config_name.split('_')[1]  # Extract config type from name
        model_name = result.get('model', 'Unknown')
        f1 = result.get('f1', 0)
        
        configs[config_key].append({
            'model': model_name,
            'f1': f1,
            'precision': result.get('precision', 0),
            'recall': result.get('recall', 0)
        })
    
    # Rank models within each configuration
    for config_key, models in configs.items():
        if models:
            # Sort by F1 score descending
            sorted_models = sorted(models, key=lambda x: x['f1'], reverse=True)
            detailed['rankings'][config_key] = [
                f"{m['model']}: F1={m['f1']:.4f}, Precision={m['precision']:.4f}, Recall={m['recall']:.4f}"
                for m in sorted_models
            ]
    
    # Calculate improvements
    url_only_f1 = None
    combined_best_f1 = None
    
    if 'url_only' in detailed['rankings'] and detailed['rankings']['url_only']:
        url_only_f1 = detailed['rankings']['url_only'][0].split('F1=')[1].split(',')[0]
    
    if 'combined' in detailed['rankings'] and detailed['rankings']['combined']:
        combined_best_f1 = detailed['rankings']['combined'][0].split('F1=')[1].split(',')[0]
    
    if url_only_f1 and combined_best_f1:
        try:
            url_only_f1 = float(url_only_f1)
            combined_best_f1 = float(combined_best_f1)
            improvement_pct = (combined_best_f1 - url_only_f1) / url_only_f1 * 100 if url_only_f1 != 0 else 0
            detailed['improvement_comparison'] = {
                'url_only_best_f1': url_only_f1,
                'combined_best_f1': combined_best_f1,
                'improvement_percent': improvement_pct
            }
        except (ValueError, TypeError):
            detailed['improvement_comparison'] = {'error': 'Could not calculate improvement'}
    else:
        detailed['improvement_comparison'] = {'note': 'Insufficient data for comparison'}
    
    # Write detailed report
    with open(REPORT_PATH, 'w') as f:
        json.dump(detailed, f, indent=2)
    
    print(f"Detailed evaluation report written to {REPORT_PATH}")
    return detailed


def print_summary():
    """Print a summary of evaluation results."""
    report = load_evaluation()
    if report is None:
        return
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    improvement = report.get('improvement', {})
    if improvement:
        url_f1 = improvement.get('url_only_f1', 0)
        combined_f1 = improvement.get('combined_f1', 0)
        imp_pct = improvement.get('improvement_percent', 0)
        print(f"\nURL-only (Logistic Regression) F1: {url_f1:.4f}")
        print(f"Combined (XGBoost) F1: {combined_f1:.4f}")
        print(f"Improvement: {imp_pct:.2f}%")
    
    print("\nConfiguration Rankings:")
    results = report.get('results', {})
    
    for config_type in ['url_only', 'content_only', 'combined']:
        print(f"\n  {config_type.upper()}:")
        config_results = [r for r in results.keys() if config_type in r]
        if config_results:
            for r in sorted(config_results, key=lambda x: results[x].get('f1', 0), reverse=True):
                result = results[r]
                print(f"    {r}: F1={result.get('f1', 0):.4f} ({result.get('model', '?')})")
        else:
            print("    No results available")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_summary()
    generate_detailed_report()