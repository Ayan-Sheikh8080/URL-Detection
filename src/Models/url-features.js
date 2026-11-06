// URL feature extraction and ML model
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

module.exports = { extractUrlFeatures };