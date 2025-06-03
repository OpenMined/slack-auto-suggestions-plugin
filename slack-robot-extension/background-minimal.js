// Minimal background script for testing
console.log('Slack Robot Extension Background Script loaded');

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Background received message:', request);
  sendResponse({ success: true, message: 'Background script working' });
});