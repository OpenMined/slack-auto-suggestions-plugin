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
      
    case 'apiRequest':
      handleApiRequest(request, sendResponse);
      return true;
      
    case 'uploadDocuments':
      handleUploadDocuments(request.files, request.uploadedBy, sendResponse);
      return true;
      
    case 'listDocuments':
      handleListDocuments(request.filters, sendResponse);
      return true;
      
    case 'searchDocuments':
      handleSearchDocuments(request.query, sendResponse);
      return true;
      
    case 'deleteDocument':
      handleDeleteDocument(request.filename, sendResponse);
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
      message: data.suggestion,  // Fix: API returns 'suggestion' not 'message'
      metadata: data.sources,    // Fix: API returns 'sources' not 'metadata'
      source: data.sources,      // Fix: API returns 'sources' not 'source'
      llm_used: data.llm_used,
      confidence: data.confidence,
      processing_time: data.processing_time
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
    
    // Transform the API response to match what the extension expects
    const transformedData = {
      ...data,
      active: data.status === 'active' // Convert status to active boolean
    };
    
    sendResponse({ 
      success: true, 
      data: transformedData 
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

// Handle generic API requests (for message capture)
async function handleApiRequest(request, sendResponse) {
  try {
    const { endpoint, method, data } = request;
    console.log(`API Request: ${method} ${endpoint}`);
    
    const options = {
      method: method || 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    };
    
    if (method === 'POST' && data) {
      options.body = JSON.stringify(data);
    }
    
    const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
    const responseData = await response.json();
    
    if (!response.ok) {
      throw new Error(responseData.detail || `HTTP error! status: ${response.status}`);
    }
    
    sendResponse({ 
      success: true, 
      data: responseData 
    });
  } catch (error) {
    console.error('API request error:', error);
    sendResponse({ 
      success: false, 
      error: error.message 
    });
  }
}

// Document upload functionality
async function handleUploadDocuments(files, uploadedBy, sendResponse) {
  try {
    console.log('Uploading documents:', files);
    
    if (files.length === 0) {
      throw new Error('No files to upload');
    }
    
    // Upload files sequentially since API accepts one file at a time
    const uploadResults = [];
    let totalChunks = 0;
    
    for (let i = 0; i < files.length; i++) {
      const fileData = files[i];
      console.log(`Uploading file ${i + 1}/${files.length}: ${fileData.name}`);
      
      // Create FormData for this file
      const formData = new FormData();
      
      // Convert base64 data to blob
      const response = await fetch(fileData.dataUrl);
      const blob = await response.blob();
      const file = new File([blob], fileData.name, { type: fileData.type });
      formData.append('file', file);
      
      // Add user_id field
      if (uploadedBy) {
        formData.append('user_id', uploadedBy);
      }
      
      const uploadResponse = await fetch(`${API_BASE_URL}/api/documents/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (!uploadResponse.ok) {
        let errorMessage = `HTTP error! status: ${uploadResponse.status}`;
        try {
          const error = await uploadResponse.json();
          // Handle different API error response formats
          if (error.detail) {
            if (typeof error.detail === 'string') {
              errorMessage = error.detail;
            } else if (Array.isArray(error.detail)) {
              // Handle Pydantic validation errors (array of error objects)
              const firstError = error.detail[0];
              if (firstError && firstError.msg) {
                errorMessage = firstError.msg;
              } else {
                errorMessage = `Validation error: ${JSON.stringify(error.detail)}`;
              }
            } else {
              errorMessage = JSON.stringify(error.detail);
            }
          } else if (error.error) {
            errorMessage = error.error;
          } else if (error.message) {
            errorMessage = error.message;
          } else if (typeof error === 'string') {
            errorMessage = error;
          } else {
            // Log the full error object for debugging
            console.error('Unknown error format from API:', error);
            errorMessage = `Upload failed with status ${uploadResponse.status}`;
          }
        } catch (parseError) {
          console.error('Could not parse error response:', parseError);
          // Keep the default HTTP error message
        }
        throw new Error(`Failed to upload ${fileData.name}: ${errorMessage}`);
      }
      
      const data = await uploadResponse.json();
      console.log(`Upload successful for ${fileData.name}:`, data);
      
      uploadResults.push({
        filename: fileData.name,
        document_id: data.document_id,
        status: 'success',
        chunks: data.processing_stats?.total_chunks || 0
      });
      
      totalChunks += data.processing_stats?.total_chunks || 0;
    }
    
    // Return aggregated results in expected format
    const response = {
      documents_processed: uploadResults.length,
      total_chunks: totalChunks,
      file_details: uploadResults
    };
    
    console.log('All uploads completed:', response);
    
    sendResponse({ 
      success: true, 
      data: response
    });
  } catch (error) {
    console.error('Error uploading documents:', error);
    sendResponse({ 
      success: false, 
      error: error.message 
    });
  }
}

async function handleListDocuments(filters, sendResponse) {
  try {
    let url = `${API_BASE_URL}/api/documents`;
    const params = new URLSearchParams();
    
    if (filters?.limit) params.append('limit', filters.limit);
    if (filters?.filename) params.append('filename', filters.filename);
    
    if (params.toString()) {
      url += '?' + params.toString();
    }
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    sendResponse({ 
      success: true, 
      data: data 
    });
  } catch (error) {
    console.error('Error listing documents:', error);
    sendResponse({ 
      success: false, 
      error: error.message 
    });
  }
}

async function handleSearchDocuments(query, sendResponse) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/documents/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(query)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    sendResponse({ 
      success: true, 
      data: data 
    });
  } catch (error) {
    console.error('Error searching documents:', error);
    sendResponse({ 
      success: false, 
      error: error.message 
    });
  }
}

async function handleDeleteDocument(filename, sendResponse) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/documents/${encodeURIComponent(filename)}`, {
      method: 'DELETE'
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    sendResponse({ 
      success: true, 
      data: data 
    });
  } catch (error) {
    console.error('Error deleting document:', error);
    sendResponse({ 
      success: false, 
      error: error.message 
    });
  }
}

// Log when extension loads
console.log('Slack Robot Extension Enhanced Background Script loaded');