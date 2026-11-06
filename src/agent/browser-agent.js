// Sandboxed browser agent for phishing URL analysis
const { chromium } = require('playwright');

/**
 * Analyze a URL in a sandboxed browser environment
 * @param {string} url - The URL to analyze
 * @returns {object} Analysis results
 */
async function analyzeUrl(url) {
  const browser = await chromium.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage']
  });
  
  const context = await browser.createContext({
    viewport: { width: 1920, height: 1080 },
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
  });
  
  const page = await context.newPage();
  
  try {
    // Navigate with timeout
    await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
    
    // Collect evidence
    const evidence = {
      redirect_chain: [],
      final_domain: null,
      forms: [],
      iframes: [],
      page_text: '',
      certificate: null
    };
    
    // Extract forms
    const forms = await page.evaluate(() => {
      const formElements = document.querySelectorAll('form');
      const forms = [];
      formElements.forEach((form, i) => {
        forms.push({
          action: form.getAttribute('action'),
          method: form.getAttribute('method'),
          inputs: Array.from(form.querySelectorAll('input')).map(input => ({
            type: input.getAttribute('type'),
            name: input.getAttribute('name'),
            value: input.getAttribute('value')
          }))
        });
      });
      return forms;
    });
    evidence.forms = forms;
    
    // Extract iframes
    const iframes = await page.evaluate(() => {
      const iframeElements = document.querySelectorAll('iframe');
      return Array.from(iframeElements).map(iframe => ({
        src: iframe.getAttribute('src'),
        name: iframe.getAttribute('name')
      }));
    });
    evidence.iframes = iframes;
    
    // Get page title and text
    const title = await page.title();
    const textContent = await page.evaluate(() => document.body.innerText || '');
    evidence.page_text = textContent.substring(0, 5000); // Limit length
    
    // Get final domain
    const finalUrl = page.url();
    evidence.final_domain = new URL(finalUrl).hostname;
    
    return {
      verdict: 'pending', // Will be set by LLM/ML model
      confidence: 0.0,
      evidence: evidence
    };
    
  } catch (error) {
    console.error('Browser analysis error:', error);
    return {
      verdict: 'error',
      confidence: 0.0,
      evidence: { redirect_chain: [], forms: [], iframes: [], page_text: '', certificate: null }
    };
  } finally {
    await browser.close();
  }
}

module.exports = { analyzeUrl };