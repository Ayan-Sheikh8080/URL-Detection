// Main entry point for the Phishing URL Analyzer API
const express = require('express');
const router = express.Router();

const app = express();
const PORT = process.env.PORT || 8000;

app.use(express.json());
app.use('/api', router);

// Health check endpoint
router.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'phishing-url-analyzer' });
});

// URL analysis endpoint
router.post('/analyze', async (req, res) => {
  const { url } = req.body;
  
  if (!url) {
    return res.status(400).json({ error: 'URL is required' });
  }
  
  try {
    // TODO: Integrate with the browser agent and ML model
    // For now, return a placeholder response
    res.json({
      verdict: 'pending',
      confidence: 0.0,
      reasons: 'Analysis queued',
      evidence: {
        redirect_chain: [],
        final_domain: null,
        forms: [],
        iframes: [],
        page_text: '',
        certificate: null
      }
    });
  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({ error: 'Analysis failed' });
  }
});

// Root endpoint
router.get('/', (req, res) => {
  res.json({ message: 'Phishing URL Analyzer API' });
});

app.listen(PORT, () => {
  console.log(`Phishing URL Analyzer running on port ${PORT}`);
});