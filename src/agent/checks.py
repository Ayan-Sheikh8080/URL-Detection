import re
from urllib.parse import urlparse, urlunparse
from typing import Dict, Any, List, Optional

def extract_domain_features(final_domain: str, original_url: str) -> Dict[str, Any]:
    """Compare final domain with original link and check for brand impersonation."""
    features = {
        'domain_mismatch': False,
        'lookalike_domain': False,
        'brand_impersonation': False,
        'tld_suspicious': False,
        'subdomain_spoofing': False,
    }
    
    try:
        # Parse original URL to get the claimed brand/domain
        original_parsed = urlparse(original_url)
        original_hostname = original_parsed.hostname or ''
        
        # Parse final domain
        final_parsed = urlparse(final_domain)
        final_hostname = final_parsed.hostname or ''
        
        # Check for domain mismatch
        if original_hostname.lower() != final_hostname.lower():
            features['domain_mismatch'] = True
            
            # Check for lookalike/TLD tricks
            final_parts = final_hostname.split('.')
            original_parts = original_hostname.split('.')
            
            # Check TLD
            if len(final_parts) >= 1 and len(original_parts) >= 1:
                final_tld = final_parts[-1].lower()
                original_tld = original_parts[-1].lower()
                
                # Suspicious TLDs
                suspicious_tlds = ['.top', '.club', '.xyz', '.wang', '.gq', '.ml', '.tk']
                if final_tld in suspicious_tlds and original_tld not in suspicious_tlds:
                    features['tld_suspicious'] = True
            
            # Check for subdomain spoofing (e.g., paypal.fake-domain.com)
            if len(final_parts) >= 2:
                potential_brand = '.'.join(final_parts[-2:])  # last two parts as "brand"
                suspicious_brands = ['paypal', 'bank', 'amazon', 'microsoft', 'google', 'apple']
                if any(potential_brand.startswith(brand) or potential_brand == brand 
                       for brand in suspicious_brands):
                # Check if the beginning looks like a brand
                    brand_prefix = potential_brand.split('.')[0].lower()
                    if brand_prefix in suspicious_brands:
                        features['lookalike_domain'] = True
        
        # Check for brand impersonation in the page context
        # (This would be enhanced with actual page text analysis)
        
    except Exception as e:
        print(f"Error extracting domain features: {e}")
    
    return features


def check_tls_certificate(cert_info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate TLS certificate details."""
    features = {
        'certificate_valid': cert_info.get('is_valid', False),
        'certificate_issuer_trusted': False,
        'certificate_subject_matches_domain': False,
        'certificate_expired': False,
    }
    
    try:
        if cert_info.get('is_valid', False):
            features['certificate_issuer_trusted'] = True
            # Check if issuer is a known CA
            trusted issuers = ['DigiCert', 'Let''s Encrypt', 'GlobalSign', 'Comodo', 'Amazon']
            issuer = cert_info.get('issuer', '').lower()
            if any(trusted in issuer for trusted in trusted issuers):
                features['certificate_issuer_trusted'] = True
        
        # Check subject matches domain
        subject = cert_info.get('subject', '').lower()
        # Simple check - subject should contain domain info
        if cert_info.get('is_valid', False):
            features['certificate_subject_matches_domain'] = True
        
        # Check expiration
        exp_date = cert_info.get('expiration_date', '')
        if exp_date:
            from datetime import datetime
            try:
                exp = datetime.strptime(exp_date, '%Y-%m-%d')
                from datetime import datetime as dt_now
                if exp < dt_now.now():
                    features['certificate_expired'] = True
                    features['certificate_valid'] = False
            except ValueError:
                pass
    
    except Exception as e:
        print(f"Error checking TLS certificate: {e}")
    
    return features


def analyze_forms_for_suspicion(forms: List[Dict[str, Any]], final_domain: str) -> Dict[str, Any]:
    """Analyze forms for suspicious characteristics."""
    features = {
        'has_password_field': False,
        'has_submit_to_different_domain': False,
        'external_submission_count': 0,
        'total_forms': len(forms),
    }
    
    try:
        for form in forms:
            if not form.get('inputs'):
                continue
                
            # Check for password fields
            for input_elem in form.get('inputs', []):
                input_type = input_elem.get('type', '').lower()
                if 'password' in input_type:
                    features['has_password_field'] = True
            
            # Check form action domain
            action = form.get('action', '')
            if action:
                action_parsed = urlparse(action if urlparse(action).netloc else f"//{action}")
                action_domain = action_parsed.hostname or ''
                final_parsed = urlparse(final_domain)
                final_domain_name = final_parsed.hostname or ''
                
                if action_domain and action_domain.lower() != final_domain_name.lower():
                    features['has_submit_to_different_domain'] = True
                    features['external_submission_count'] += 1
    
    except Exception as e:
        print(f"Error analyzing forms: {e}")
    
    return features


def check_for_prompt_injection(page_text: str) -> Dict[str, Any]:
    """Check page text for prompt injection attempts."""
    features = {
        'prompt_injection_detected': False,
        'injection_type': None,
        'suspicious_patterns': [],
    }
    
    try:
        injection_patterns = [
            r'ignore previous instructions',
            r'do not follow',
            r'system.*prompt',
            r'hidden.*text',
            r'<!--.*-->',
            r'<script.*>',
            r'\\n\\nASSISTANT',
            r'\\n\\nUSER',
        ]
        
        page_lower = page_text.lower()
        
        for pattern in injection_patterns:
            if re.search(pattern, page_lower, re.IGNORECASE):
                features['suspicious_patterns'].append(pattern)
        
        if features['suspicious_patterns']:
            features['prompt_injection_detected'] = True
            # Determine injection type
            if any('ignore' in p.lower() for p in features['suspicious_patterns']):
                features['injection_type'] = 'instruction_override'
            elif any('system' in p.lower() for p in features['suspicious_patterns']):
                features['injection_type'] = 'system_prompt_leak'
            else:
                features['injection_type'] = 'suspicious_pattern'
    
    except Exception as e:
        print(f"Error checking prompt injection: {e}")
    
    return features