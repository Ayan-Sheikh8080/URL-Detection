// FastAPI endpoint for phishing analysis
const express = require('express');
const router = express.Router();
const { extractUrlFeatures } = require('../models/url-features');
const { analyzeUrl } = require('../agent/browser-agent');

router.post('/', async (req, res) => {
  const { url } = req.body;
  
  if (!url) {
    return res.status(400).json({ error: 'URL is required' });
  }
  
  try {
    // Step 1: Extract URL features
    const urlFeatures = extractUrlFeatures(url);
    
    // Step 2: Run ML model prediction
    let mlPrediction = { verdict: 'unknown', confidence: 0.0, score: 0.0 };
    if (urlFeatures) {
      mlPrediction = await predictUrl(urlFeatures);
    }
    
    // Step 3: Run browser agent analysis
    const browserEvidence = await analyzeUrl(url);
    
    // Step 4: Combine evidence (simplified - will use LLM in full implementation)
    const combinedVerdict = combineEvidence(mlPrediction, browserEvidence);
    
    res.json({
      url,
      verdict: combinedVerdict.verdict,
      confidence: combinedVerdict.confidence,
      reasons: combinedVerdict.reasons,
      ml_score: mlPrediction.score,
      evidence: browserEvidence.evidence
    });
  } catch (error) {
    console.error('API error:', error);
    res.status(500).json({ error: 'Analysis failed' });
  }
});

function combineEvidence(mlPrediction, browserEvidence) {
  // Simple combination logic - will be replaced by LLM reasoning
  const mlVerdict = mlPrediction.verdict || 'unknown';
  const evidence = browserEvidence.evidence;
  
  let finalVerdict = 'safe';
  let confidence = 0.5;
  let reasons = 'ML model and browser evidence combined';
  
  // If either model or evidence indicates suspicious, flag it
  if (mlVerdict === 'phishing' || evidence.forms.length > 0) {
    finalVerdict = 'suspicious';
    confidence = 0.7;
    reasons = 'URL features or browser evidence indicated suspicious activity';
  }
  
  return { verdict: finalVerdict, confidence, reasons };
}

module.exports = router;