// Enhanced background script with LLM provider management
const API_BASE_URL = 'http://localhost:8000';

// Handle messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Background received message:', request);
  
  switch (request.action) {
    case 'fetchSuggestion':
      handleFetchSuggestion(request.messageData, sendResponse);
      return true; // Will respond asynchronously
      
    case 'getCurrentProvider':
      handleGetCurrentProvider(sendResponse);
      return true;
      
    case 'getProviderInfo':
      handleGetProviderInfo(sendResponse);
      return true;
      
    case 'configureProvider':
      handleConfigureProvider(request.config, sendResponse);
      return true;
      
    case 'activateProvider':
      handleActivateProvider(request.provider, sendResponse);
      return true;
      
    case 'testProvider':
      handleTestProvider(sendResponse);
      return true;
      
    default:
      sendResponse({ success: false, error: 'Unknown action' });
  }
});

// Fetch suggestion with enhanced context
async function handleFetchSuggestion(messageData, sendResponse) {
  try {
    console.log('Fetching suggestion with enhanced data:', messageData);
    
    // Build request body matching the SuggestionRequest format
    const requestBody = {
      message: messageData.content || 'Hello',
      conversation_history: messageData.conversation_history || [],
      user_id: messageData.user_id || 'default',
      channel_id: messageData.conversation_id || 'general',
      thread_id: messageData.thread_ts || null,
      max_tokens: 500,
      temperature: 0.7,
      include_sources: true,
      use_knowledge_graph: true
    };
    
    const response = await fetch(`${API_BASE_URL}/api/suggestion`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Received suggestion response:', data);
    
    sendResponse({ 
      success: true, 
      message: data.message,
      metadata: data.metadata,
      source: data.source,
      llm_used: data.llm_used
    });
  } catch (error) {
    console.error('Error fetching suggestion:', error);
    sendResponse({ 
      success: false, 
      message: `Failed to fetch suggestion: ${error.message}. Make sure the API server is running on ${API_BASE_URL}` 
    });
  }
}

// Get current LLM provider info
async function handleGetCurrentProvider(sendResponse) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/llm/current`);
    const data = await response.json();
    
    sendResponse({ 
      success: true, 
      data: data 
    });
  } catch (error) {
    console.error('Error getting current provider:', error);
    sendResponse({ 
      success: false, 
      error: error.message 
    });
  }
}

// Get all provider information
async function handleGetProviderInfo(sendResponse) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/llm/providers`);
    const data = await response.json();
    
    sendResponse({ 
      success: true,
      providers: data.providers || [],
      current: data.current || {},
      available_providers: data.available_providers || []
    });
  } catch (error) {
    console.error('Error getting provider info:', error);
    sendResponse({ 
      success: false, 
      error: error.message 
    });
  }
}

// Configure a provider
async function handleConfigureProvider(config, sendResponse) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/llm/providers/configure`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config)
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Configuration failed');
    }
    
    const data = await response.json();
    sendResponse({ 
      success: true, 
      data: data 
    });
  } catch (error) {
    console.error('Error configuring provider:', error);
    sendResponse({ 
      success: false, 
      error: error.message 
    });
  }
}

// Activate a provider
async function handleActivateProvider(provider, sendResponse) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/llm/providers/activate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ provider })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Activation failed');
    }
    
    const data = await response.json();
    sendResponse({ 
      success: true, 
      data: data 
    });
  } catch (error) {
    console.error('Error activating provider:', error);
    sendResponse({ 
      success: false, 
      error: error.message 
    });
  }
}

// Test current provider
async function handleTestProvider(sendResponse) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/llm/providers/test`, {
      method: 'POST'
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Test failed');
    }
    
    const data = await response.json();
    
    // Extract the response text from the test
    const testResponse = data.response || 
                        data.checks?.connectivity?.response || 
                        'Test completed';
    
    sendResponse({ 
      success: true, 
      response: testResponse,
      data: data 
    });
  } catch (error) {
    console.error('Error testing provider:', error);
    sendResponse({ 
      success: false, 
      error: error.message 
    });
  }
}

// Log when extension loads
console.log('Slack Robot Extension Enhanced Background Script loaded');