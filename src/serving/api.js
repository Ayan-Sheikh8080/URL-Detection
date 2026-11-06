from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import time

class Verdict(Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    PHISHING = "phishing"

@dataclass
class AnalysisRequest:
    """Request schema for URL analysis."""
    url: str
    # Optional overrides
    model_config: str = "combined"  # url_only, content_only, combined
    use_llm: bool = False
    
    def validate(self) -> bool:
        """Validate the request."""
        if not self.url or not self.url.strip():
            return False
        # Basic URL validation
        from urllib.parse import urlparse
        try:
            result = urlparse(self.url)
            return all([result.scheme, result.netloc])
        except:
            return False

@dataclass
class EvidenceSummary:
    """Collected evidence from browser sandbox."""
    redirect_chain: List[str] = field(default_factory=list)
    final_domain: str = ""
    forms: List[Dict[str, Any]] = field(default_factory=list)
    iframes: List[Dict[str, Any]] = field(default_factory=list)
    page_text: str = ""
    title: str = ""
    certificate: Dict = field(default_factory=dict)

@dataclass 
class AnalysisResponse:
    """Response schema for URL analysis."""
    url: str
    verdict: Verdict
    confidence: float
    reasons: List[str] = field(default_factory=list)
    model_score: float = 0.0
    combined_score: float = 0.0
    top_factors: List[Dict[str, str]] = field(default_factory=list)
    evidence_summary: EvidenceSummary = field(default_factory=EvidenceSummary)
    browser_ok: bool = True
    load_ms: float = 0.0
    reasoner: str = "rule_based"

@dataclass
class HealthResponse:
    """Response for health check."""
    status: str = "ok"
    service: str = "phishing-url-analyzer"
    version: str = "1.0.0"
    uptime_ms: float = 0.0

@dataclass
class MonitoringStats:
    """Monitoring statistics."""
    total_analyses: int = 0
    verdict_distribution: Dict[str, int] = field(default_factory=lambda: {"safe": 0, "suspicious": 0, "phishing": 0})
    average_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    average_model_score: float = 0.0
    browser_failure_rate: float = 0.0
```

Now let me create the FastAPI-compatible Express API:
<tool_call>
<function=write>
<parameter=content>
const express = require('express');
const router = express.Router();
const { extractUrlFeatures } = require('../models/url-features');
const { analyzeUrl } = require('../agent/browser-agent');
const { get_reasoner } = require('../agent/reasoning');
const { ModelRegistry } = require('../ml/model_registry');

const router = express.Router();
const modelRegistry = new ModelRegistry();

// Track monitoring stats (in-memory for now; would use Redis/DynamoDB in production)
let monitoringStats = {
    total_analyses: 0,
    verdict_distribution: { safe: 0, suspicious: 0, phishing: 0 },
    total_latency: 0,
    latency_samples: [],
    average_model_score: 0,
    browser_failures: 0
};

/**
 * POST /analyze - Analyze a URL for phishing
 */
router.post('/analyze', async (req, res) => {
  const startTime = Date.now();
  
  const { url, model_config = 'combined', use_llm = false } = req.body;
  
  if (!url) {
    return res.status(400).json({ error: 'URL is required' });
  }
  
  try {
    // Step 1: Extract URL features
    const urlFeatures = extractUrlFeatures(url);
    
    // Step 2: Run ML model prediction
    let mlPrediction = { score: 0.5, verdict: 'unknown' };
    if (urlFeatures) {
      mlPrediction = await predictUrl(urlFeatures);
    }
    
    // Step 3: Run browser agent analysis
    const browserResult = await analyzeUrl(url);
    
    // Step 4: Combine evidence using reasoning layer
    const reasoner = get_reasoner();  // Uses rule-based by default
    
    const reasoningInput = {
      url,
      url_model_score: mlPrediction.score,
      evidence: browserResult.evidence,
      page_text: browserResult.evidence.page_text,
      certificate_info: browserResult.evidence.certificate 
        ? { is_valid: browserResult.evidence.certificate.is_valid }
        : { is_valid: true }
    };
    
    const reasoningOutput = reasoner.reason(reasoningInput);
    
    // Step 5: Update monitoring stats
    const loadMs = Date.now() - startTime;
    monitoringStats.total_analyses++;
    monitoringStats.total_latency += loadMs;
    monitoringStats.latency_samples.push(loadMs);
    
    // Keep only last 1000 samples
    if (monitoringStats.latency_samples.length > 1000) {
      monitoringStats.latency_samples = monitoringStats.latency_samples.slice(-1000);
    }
    
    // Update verdict distribution
    const verdictKey = reasoningOutput.verdict === 'phishing' ? 'phishing' 
                      : reasoningOutput.verdict === 'suspicious' ? 'suspicious' : 'safe';
    if (monitoringStats.verdict_distribution[verdictKey] !== undefined) {
      monitoringStats.verdict_distribution[verdictKey]++;
    }
    
    // Update model score average
    monitoringStats.average_model_score = 
      (monitoringStats.average_model_score * (monitoringStats.total_analyses - 1) + mlPrediction.score) 
      / monitoringStats.total_analyses;
    
    // Update browser failures
    if (!browserResult.browser_ok) {
      monitoringStats.browser_failures++;
    }
    
    // Build response
    const response: AnalysisResponse = {
      url,
      verdict: reasoningOutput.verdict,
      confidence: reasoningOutput.confidence,
      reasons: reasoningOutput.reasons,
      model_score: mlPrediction.score,
      combined_score: reasoningOutput.combined_score,
      top_factors: reasoningOutput.top_factors,
      evidence_summary: browserResult.evidence,
      browser_ok: browserResult.browser_ok,
      load_ms: loadMs,
      reasoner: reasoningOutput.reasoner
    };
    
    res.json(response);
    
  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({ error: 'Analysis failed', details: error.message });
  }
});

/**
 * GET /healthz - Health check endpoint
 */
router.get('/healthz', (req, res) => {
  const healthResponse: HealthResponse = {
    status: 'ok',
    service: 'phishing-url-analyzer',
    version: '1.0.0',
    uptime_ms: process.uptime() * 1000
  };
  res.json(healthResponse);
});

/**
 * GET /monitoring/stats - Get monitoring statistics
 */
router.get('/monitoring/stats', (req, res) => {
  // Calculate percentiles from latency samples
  const sortedSamples = [...monitoringStats.latency_samples].sort((a, b) => a - b);
  const p50Index = Math.floor(sortedSamples.length * 0.5);
  const p95Index = Math.floor(sortedSamples.length * 0.95);
  
  const stats: MonitoringStats = {
    total_analyses: monitoringStats.total_analyses,
    verdict_distribution: { ...monitoringStats.verdict_distribution },
    average_latency_ms: monitoringStats.total_analyses > 0 
      ? monitoringStats.total_latency / monitoringStats.total_analyses 
      : 0,
    p50_latency_ms: sortedSamples[p50Index] || 0,
    p95_latency_ms: sortedSamples[p95Index] || 0,
    max_latency_ms: Math.max(...sortedSamples) || 0,
    average_model_score: monitoringStats.average_model_score,
    browser_failure_rate: monitoringStats.total_analyses > 0
      ? (monitoringStats.browser_failures / monitoringStats.total_analyses) * 100
      : 0
  };
  
  res.json(stats);
});

/**
 * GET /monitoring/drift - Get drift report
 */
router.get('/monitoring/drift', (req, res) => {
  // Check if reference features are available
  const referenceFeatures = modelRegistry.get_reference_features();
  
  if (!referenceFeatures) {
    return res.json({
      drift_detected: false,
      message: 'No reference features available for drift comparison',
      psi_threshold: 0.5
    });
  }
  
  // In a full implementation, this would compare recent feature distributions
  # with reference features using Population Stability Index (PSI)
  # For now, return placeholder
  res.json({
    drift_detected: false,
    message: 'Drift monitoring not implemented in this version',
    psi_threshold: 0.5,
    features_monitored: modelRegistry.feature_schema?.feature_order?.length || 0
  });
});

/**
 * GET / - Root endpoint
 */
router.get('/', (req, res) => {
  res.json({ 
    message: 'Phishing URL Analyzer API',
    version: '1.0.0',
    endpoints: ['/analyze', '/healthz', '/monitoring/stats', '/monitoring/drift']
  });
});

module.exports = router;