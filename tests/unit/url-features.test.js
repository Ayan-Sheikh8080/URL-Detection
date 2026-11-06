// Unit tests for the phishing URL analyzer
const { extractUrlFeatures } = require('../../src/Models/url-features');

describe('URL Feature Extraction', () => {
  test('extracts features from a valid URL', () => {
    const url = 'https://example.com/path';
    const features = extractUrlFeatures(url);
    expect(features).toBeDefined();
    expect(features.url_length).toBe(url.length);
    expect(features.https).toBe(1);
  });

  test('detects IP addresses in URLs', () => {
    const url = 'http://192.168.1.1/path';
    const features = extractUrlFeatures(url);
    expect(features.has_ip_address).toBe(1);
  });

  test('detects HTTPS protocol', () => {
    const url = 'http://example.com/path';
    const features = extractUrlFeatures(url);
    expect(features.https).toBe(0);
  });
});