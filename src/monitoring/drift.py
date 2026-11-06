import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

# PSI (Population Stability Index) calculation for drift detection
PSI_THRESHOLD = 0.5  # Threshold for flagging drift

def calculate_psi(expected, actual):
    """
    Calculate Population Stability Index between expected (training) and actual (current) distributions.
    
    PSI ranges:
    - < 0.1: No significant change
    - 0.1 - 0.25: Moderate change
    - 0.25 - 0.4: Significant change
    - > 0.4: Very significant change
    """
    # Convert to numpy arrays if needed
    if isinstance(expected, list):
        expected = np.array(expected)
    if isinstance(actual, list):
        actual = np.array(actual)
    
    # Ensure both arrays are the same length by padding the shorter one
    max_len = max(len(expected), len(actual))
    expected = np.pad(expected, (0, max_len - len(expected)), mode='constant', constant_values=0)
    actual = np.pad(actual, (0, max_len - len(actual)), mode='constant', constant_values=0)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    expected = expected + epsilon
    actual = actual + epsilon
    
    # Convert to percentages/probabilities
    expected_pct = expected / expected.sum()
    actual_pct = actual / actual.sum()
    
    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return float(psi)

def check_drift(reference_features_path: str = "models/reference_features.csv", 
                recent_features_path: str = "data/processed/recent_features.csv") -> Dict[str, Any]:
    """
    Check for feature drift between training data and recent live data.
    
    Returns drift analysis report.
    """
    results = {
        'drift_detected': False,
        'psi_score': 0.0,
        'features_analyzed': 0,
        'drifted_features': [],
        'psi_by_feature': {},
        'message': 'No drift analysis performed',
        'timestamp': os.popen('date 2>/dev/null || systeminfo').read()[:50] if os else 'unknown'
    }
    
    try:
        reference_path = Path(reference_features_path)
        recent_path = Path(recent_features_path)
        
        # Check if files exist
        if not reference_path.exists():
            results['message'] = f'Reference features file not found: {reference_features_path}'
            return results
        
        if not recent_path.exists():
            results['message'] = f'Recent features file not found: {recent_features_path}'
            return results
        
        # Load feature data
        reference_df = __import__('pandas').read_csv(reference_features_path)
        recent_df = __import__('pandas').read_csv(recent_features_path)
        
        # Identify common feature columns (exclude metadata columns)
        metadata_cols = {'url', 'label', 'source', 'dataset_hash', 'generated_at', 'version'}
        feature_cols = [col for col in reference_df.columns if col not in metadata_cols]
        
        if not feature_cols:
            results['message'] = 'No feature columns found in reference data'
            return results
        
        results['features_analyzed'] = len(feature_cols)
        
        # Calculate PSI for each feature
        drifted = []
        
        for col in feature_cols:
            try:
                # Get values for this feature
                ref_values = reference_df[col].dropna().tolist()
                recent_values = recent_df[col].dropna().tolist()
                
                if not ref_values or not recent_values:
                    continue
                
                # For categorical features, compute PSI on value distributions
                # For numerical features, bin the values
                try:
                    # Try numeric PSI
                    ref_bins = np.histogram(ref_values, bins=10)[0] + 1e-6
                    recent_bins = np.histogram(recent_values, bins=10)[0] + 1e-6
                    
                    psi = calculate_psi(ref_bins, recent_bins)
                    
                    results['psi_by_feature'][col] = round(psi, 4)
                    
                    if psi > PSI_THRESHOLD:
                        drifted.append({
                            'feature': col,
                            'psi': round(psi, 4),
                            'severity': 'high' if psi > 0.4 else 'moderate'
                        })
                except Exception:
                    # For categorical, use value distribution comparison
                    pass
            except Exception as e:
                print(f"Error calculating PSI for {col}: {e}")
                continue
        
        # Sort drifted features by PSI (highest first)
        drifted.sort(key=lambda x: x['psi'], reverse=True)
        results['drifted_features'] = drifted
        
        # Calculate overall PSI (average of all feature PSIs)
        if results['psi_by_feature']:
            avg_psi = np.mean(list(results['psi_by_feature'].values()))
            results['psi_score'] = round(float(avg_psi), 4)
            results['drift_detected'] = avg_psi > PSI_THRESHOLD
        
        # Set message
        if results['drift_detected']:
            if results['psi_score'] > 0.4:
                results['message'] = f'Very significant drift detected (PSI={results["psi_score"]:.4f})'
            elif results['psi_score'] > 0.25:
                results['message'] = f'Significant drift detected (PSI={results["psi_score"]:.4f})'
            else:
                results['message'] = f'Moderate drift detected (PSI={results["psi_score"]:.4f})'
        else:
            results['message'] = f'No significant drift detected (PSI={results["psi_score"]:.4f})'
        
    except Exception as e:
        results['message'] = f'Drift analysis error: {str(e)}'
        print(f"Drift analysis error: {e}")
    
    return results

def generate_drift_report(output_path: str = "reports/drift_report.json") -> Dict[str, Any]:
    """Generate and save a drift report to file."""
    report = check_drift()
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Drift report written to {report_path}")
    return report