// URL feature extraction and ML model
const { PythonShell } = require('python-shell');

/**
 * Extract features from a URL for ML classification
 * @param {string} url - The URL to extract features from
 * @returns {object} URL features
 */
function extractUrlFeatures(url) {
  try {
    const parsedUrl = new URL(url);
    const features = {
      url_length: url.length,
      https: parsedUrl.protocol === 'https:' ? 1 : 0,
      domain_length: parsedUrl.hostname.length,
      path_length: parsedUrl.pathname.length,
      symbol_count: (url.match(/[^a-zA-Z0-9]/g) || []).length,
      has_ip_address: /^\d+\.\d+\.\d+\.\d+$/.test(parsedUrl.hostname) ? 1 : 0,
      at_symbol: url.includes('@') ? 1 : 0,
      double_slash: url.includes('//') && !url.startsWith('https://') ? 1 : 0,
      subdomain_count: (parsedUrl.hostname.split('.').length - 2) || 0
    };
    return features;
  } catch (e) {
    console.error('URL feature extraction error:', e);
    return null;
  }
}

/**
 * Run ML model prediction on URL features
 * @param {object} features - URL features extracted
 * @returns {object} Prediction result
 */
async function predictUrl(features) {
  // Use Python Shell to run the ML model
  return new Promise((resolve, reject) => {
    PythonShell.run('models/predict.py', { features: JSON.stringify(features) }, (errors, results) => {
      if (errors) {
        reject(errors);
      } else {
        try {
          resolve(JSON.parse(results[0]));
        } catch (e) {
          reject(e);
        }
      }
    });
  });
}

module.exports = { extractUrlFeatures, predictUrl };