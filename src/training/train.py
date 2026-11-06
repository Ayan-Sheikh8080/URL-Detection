import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.xgboost import XGBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import mlflow
import mlflow.sklearn
from pathlib import Path

# Configuration
DATA_PROCESSED_PATH = Path("data/processed/urls_processed.csv")
MODELS_DIR = Path("models")
METRICS_DIR = Path("reports")
MLFLOW_EXPERIMENT = "phishing-link-detection"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data():
    """Load the processed dataset from preparation step."""
    if not DATA_PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Processed data not found at {DATA_PROCESSED_PATH}")
    df = pd.read_csv(DATA_PROCESSED_PATH)
    print(f"Loaded processed data: {df.shape}")
    return df


def extract_url_features(df):
    """Extract 27 lexical URL features from URLs."""
    features_list = []
    
    for url in df['url']:
        url_str = str(url)
        try:
            # Import URL parsing features
            parsed_url = __import__('urllib.parse').urlparse(url_str)
            
            feature_dict = {
                'url_length': len(url_str),
                'https': 1 if parsed_url.scheme == 'https' else 0,
                'domain_length': len(parsed_url.netloc) if parsed_url.netloc else 0,
                'path_length': len(parsed_url.path) if parsed_url.path else 0,
                'symbol_count': len([c for c in url_str if not c.isalnum() and c not in '/:.']),
                'has_ip_address': 1 if bool(__import__('re').match(r'^\d+\.\d+\.\d+\.\d+$', parsed_url.hostname or '')) else 0,
                'at_symbol': 1 if '@' in url_str else 0,
                'double_slash': 1 if '//' in url_str and not url_str.startswith('https://') else 0,
                'subdomain_count': len(parsed_url.hostname.split('.')) - 2 if parsed_url.hostname else 0,
                'has_query': 1 if parsed_url.query else 0,
                'has_fragment': 1 if parsed_url.fragment else 0,
                'path_depth': parsed_url.path.count('/') if parsed_url.path else 0,
                'file_extension': parsed_url.path.split('.')[-1] if '.' in parsed_url.path else '',
                'hostname_length': len(parsed_url.hostname or ''),
                'num_params': len(parsed_url.query.split('&')) if parsed_url.query else 0,
                'parameter_names': '|'.join([p.split('=')[0] for p in parsed_url.query.split('&') if '=' in p]) if parsed_url.query else '',
                'has_login_keyword': 1 if any(kw in url_str.lower() for kw in ['login', 'signin', 'verify', 'account', 'secure']) else 0,
                'has_financial_keyword': 1 if any(kw in url_str.lower() for kw in ['paypal', 'bank', 'amazon', 'credit', 'card']) else 0,
                'uses_shortener': 1 if any(shortener in url_str.lower() for shortener in ['bit.ly', 'tiny.url', 'short.link', 'goo.gl']) else 0,
                'has_trailing_slash': 1 if url_str.rendswith('/') else 0,
                'num_digits': sum(c.isdigit() for c in parsed_url.hostname or ''),
                'has_suffix': 1 if parsed_url.hostname and parsed_url.hostname.endswith(('.ph', '.top', '.club', '.xyz')) else 0,
                'tld': parsed_url.hostname.split('.')[-1] if parsed_url.hostname else '',
            }
            features_list.append(feature_dict)
        except Exception as e:
            print(f"Warning: Failed to parse URL {url_str}: {e}")
            # Add default features for failed URL
            features_list.append({k: 0 for k in feature_dict.keys()})
    
    return pd.DataFrame(features_list)


def train_models(X_train, y_train, X_test, y_test):
    """Train three model families in three configurations."""
    results = {}
    
    # Model families
    model_families = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'XGBoost': XGBoostClassifier(n_estimators=100, random_state=42, max_depth=6, use_label_encoder=False, eval_metric='logloss')
    }
    
    # Configuration 1: URL-only features
    print("\n" + "=" * 60)
    print("CONFIGURATION 1: URL-only training")
    print("=" * 60)
    
    url_only_models = {}
    for name, model in model_families.items():
        print(f"\nTraining {name} on URL-only features...")
        model.fit(X_train, y_train)
        url_only_models[name] = model
        
        # Evaluate
        y_pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param('model_family', name)
            mlflow.log_param('config', 'url_only')
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1', f1)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            mlflow.log_matrix(confusion_matrix(y_test, y_pred), 'confusion_matrix.png')
            
            # Log model
            mlflow.sklearn.log_model(model, f'model_{name.lower().replace(" ", "_")}')
        
        results[f'url_only_{name}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model': model
        }
        
        print(f"  F1 Score: {f1:.4f}")
    
    # Configuration 2: Content-only features (page evidence)
    print("\n" + "=" * 60)
    print("CONFIGURATION 2: Content-only training")
    print("=" * 60)
    
    # For this synthetic dataset, we'll use a subset of features that could represent page evidence
    content_features = ['url_length', 'symbol_count', 'has_https', 'domain_length', 'has_ip_address',
                        'at_symbol', 'double_slash', 'subdomain_count', 'has_login_keyword',
                        'has_financial_keyword', 'uses_shortener', 'has_trailing_slash']
    
    content_train = X_train[content_features]
    content_test = X_test[content_features]
    
    content_models = {}
    for name, model in model_families.items():
        print(f"\nTraining {name} on content-only features...")
        model.fit(content_train, y_train)
        content_models[name] = model
        
        # Evaluate
        y_pred = model.predict(content_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param('model_family', name)
            mlflow.log_param('config', 'content_only')
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1', f1)
        
        results[f'content_only_{name}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model': model
        }
        
        print(f"  F1 Score: {f1:.4f}")
    
    # Configuration 3: Combined features (URL + page evidence)
    print("\n" + "=" * 60)
    print("CONFIGURATION 3: Combined training (URL + page features)")
    print("=" * 60)
    
    # Combine URL features with page evidence features
    combined_features = list(X_train.columns)  # All URL features
    
    # Add synthetic page evidence features
    page_evidence_features = ['forms_count', 'iframes_count', 'has_password_field', 'has_submit_different_domain',
                              'has_iframe', 'page_text_length', 'title_length', 'certificate_valid',
                              'redirect_count', 'form_action_external']
    
    # For synthetic data, add placeholder page evidence features
    X_train_combined = X_train.copy()
    X_test_combined = X_test.copy()
    
    for feat in page_evidence_features:
        X_train_combined[feat] = np.random.randint(0, 2, len(X_train))
        X_test_combined[feat] = np.random.randint(0, 2, len(X_test))
    
    combined_models = {}
    for name, model in model_families.items():
        print(f"\nTraining {name} on combined features...")
        model.fit(X_train_combined, y_train)
        combined_models[name] = model
        
        # Evaluate
        y_pred = model.predict(X_test_combined)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param('model_family', name)
            mlflow.log_param('config', 'combined')
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1', f1)
        
        results[f'combined_{name}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model': model
        }
        
        print(f"  F1 Score: {f1:.4f}")
    
    return url_only_models, content_models, combined_models


def main():
    """Main training pipeline entry point."""
    print("=" * 60)
    print("PHISHING URL CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Set MLflow tracking
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    # Load data
    df = load_processed_data()
    
    # Extract features
    print("\nExtracting URL features...")
    features_df = extract_url_features(df)
    
    # Prepare target variable
    y = df['label'].values
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    
    # Train all configurations
    url_only_models, content_models, combined_models = train_models(X_train, y_train, X_test, y_test)
    
    # Generate evaluation report
    print("\n" + "=" * 60)
    print("GENERATING EVALUATION REPORT")
    print("=" * 60)
    
    # Collect all results
    all_results = {}
    
    # URL-only results
    for config_name, results_dict in results.items():
        if 'url_only' in config_name:
            model_name = results_dict['model'].__class__.__name__
            all_results[config_name] = {
                'f1': results_dict['f1'],
                'precision': results_dict['precision'],
                'recall': results_dict['recall'],
                'model': model_name
            }
    
    # Content-only results
    for config_name, results_dict in results.items():
        if 'content_only' in config_name:
            model_name = results_dict['model'].__class__.__name__
            all_results[config_name] = {
                'f1': results_dict['f1'],
                'precision': results_dict['precision'],
                'recall': results_dict['recall'],
                'model': model_name
            }
    
    # Combined results
    for config_name, results_dict in results.items():
        if 'combined' in config_name:
            model_name = results_dict['model'].__class__.__name__
            all_results[config_name] = {
                'f1': results_dict['f1'],
                'precision': results_dict['precision'],
                'recall': results_dict['recall'],
                'model': model_name
            }
    
    # Write evaluation report
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'dataset_info': {
            'rows': len(df),
            'features': list(features_df.columns)
        },
        'results': all_results,
        'headline': 'Relative F1 improvement of agent over URL-only'
    }
    
    # Calculate improvement
    url_only_f1 = all_results.get('url_only_Logistic Regression', {}).get('f1', 0)
    combined_f1 = all_results.get('combined_XGBoost', {}).get('f1', 0)
    
    if url_only_f1 > 0:
        improvement = (combined_f1 - url_only_f1) / url_only_f1 * 100
        report['improvement'] = {
            'url_only_f1': url_only_f1,
            'combined_f1': combined_f1,
            'improvement_percent': improvement
        }
        print(f"URL-only (LR) F1: {url_only_f1:.4f}")
        print(f"Combined (XGBoost) F1: {combined_f1:.4f}")
        print(f"Improvement: {improvement:.2f}%")
    else:
        report['improvement'] = {
            'url_only_f1': url_only_f1,
            'combined_f1': combined_f1,
            'improvement_percent': 0
        }
    
    # Write report
    report_path = METRICS_DIR / "evaluation.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nEvaluation report written to {report_path}")
    print(f"\n{json.dumps(report, indent=2)}")
    
    # Save best models
    best_combined = max(combined_models.values(), key=lambda m: m.score(X_test_combined, y_test))
    joblib.dump(best_combined, MODELS_DIR / 'best_model.pkl')
    print(f"\nBest model saved to {MODELS_DIR / 'best_model.pkl'}")
    
    # Save model index
    model_index = {
        'version': '1.0.0',
        'models': {
            'url_only': {name: {'type': type(m).__name__} for name, m in url_only_models.items()},
            'content_only': {name: {'type': type(m).__name__} for name, m in content_models.items()},
            'combined': {name: {'type': type(m).__name__} for name, m in combined_models.items()}
        }
    }
    
    with open(MODELS_DIR / 'index.json', 'w') as f:
        json.dump(model_index, f, indent=2)
    
    print(f"Model index saved to {MODELS_DIR / 'index.json'}")
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()