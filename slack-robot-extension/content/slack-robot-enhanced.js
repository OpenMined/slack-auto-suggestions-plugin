(function() {
  'use strict';

  let observer = null;
  let checkInterval = null;
  let currentProvider = null;
  let providerStatus = 'unknown';

  // Icons
  function createRobotIcon() {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 20 20');
    svg.setAttribute('aria-hidden', 'true');
    svg.innerHTML = `
      <path fill="currentColor" fill-rule="evenodd" 
        d="M10 1.5a1 1 0 100 2 1 1 0 000-2zM7.5 3a2.5 2.5 0 115 0h2.25a.75.75 0 01.75.75v4.5a.75.75 0 01-.75.75h-.5v1.25h1.25a.75.75 0 01.75.75v5.25a.75.75 0 01-.75.75h-11a.75.75 0 01-.75-.75V11a.75.75 0 01.75-.75H5.5V9h-.5a.75.75 0 01-.75-.75v-4.5A.75.75 0 015 3h2.5zM5.75 4.5v3h8.5v-3h-8.5zM7 9v1.25h6V9H7zm-1.75 2.75v3.75h9.5v-3.75h-9.5z"
        clip-rule="evenodd"/>
    `;
    return svg;
  }

  function createSettingsIcon() {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 20 20');
    svg.setAttribute('aria-hidden', 'true');
    svg.innerHTML = `
      <path fill="currentColor" fill-rule="evenodd" 
        d="M7.84 1.804A1 1 0 018.82 1h2.36a1 1 0 01.98.804l.331 1.652a6.993 6.993 0 011.929 1.115l1.598-.54a1 1 0 011.186.447l1.18 2.044a1 1 0 01-.205 1.251l-1.267 1.113a7.047 7.047 0 010 2.228l1.267 1.113a1 1 0 01.206 1.25l-1.18 2.045a1 1 0 01-1.187.447l-1.598-.54a6.993 6.993 0 01-1.929 1.115l-.33 1.652a1 1 0 01-.98.804H8.82a1 1 0 01-.98-.804l-.331-1.652a6.993 6.993 0 01-1.929-1.115l-1.598.54a1 1 0 01-1.186-.447l-1.18-2.044a1 1 0 01.205-1.251l1.267-1.114a7.05 7.05 0 010-2.227L1.821 7.773a1 1 0 01-.206-1.25l1.18-2.045a1 1 0 011.187-.447l1.598.54A6.993 6.993 0 017.51 3.456l.33-1.652zM10 13a3 3 0 100-6 3 3 0 000 6z"
        clip-rule="evenodd"/>
    `;
    return svg;
  }

  function createStatusIndicator(status) {
    const indicator = document.createElement('span');
    indicator.style.cssText = `
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-left: 4px;
      vertical-align: middle;
    `;
    
    switch(status) {
      case 'active':
        indicator.style.backgroundColor = '#0f0';
        indicator.title = 'LLM Provider Active';
        break;
      case 'error':
        indicator.style.backgroundColor = '#f00';
        indicator.title = 'LLM Provider Error';
        break;
      case 'loading':
        indicator.style.backgroundColor = '#ff0';
        indicator.title = 'LLM Provider Loading';
        break;
      default:
        indicator.style.backgroundColor = '#888';
        indicator.title = 'LLM Provider Unknown';
    }
    
    return indicator;
  }

  // LLM Provider Management
  async function fetchCurrentProvider() {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ action: 'getCurrentProvider' }, (response) => {
        if (response && response.success) {
          currentProvider = response.data;
          providerStatus = response.data.active ? 'active' : 'inactive';
          updateProviderStatus();
          resolve(response.data);
        } else {
          providerStatus = 'error';
          updateProviderStatus();
          resolve(null);
        }
      });
    });
  }

  function updateProviderStatus() {
    // Update all status indicators
    const indicators = document.querySelectorAll('.robot-llm-status');
    indicators.forEach(indicator => {
      const newIndicator = createStatusIndicator(providerStatus);
      indicator.replaceWith(newIndicator);
      newIndicator.className = 'robot-llm-status';
    });
  }

  // UI Creation
  function createRobotButton(context) {
    const button = document.createElement('button');
    button.className = 'c-button-unstyled c-icon_button c-icon_button--size_small c-icon_button--default';
    
    if (context === 'sticky_bar') {
      button.className += ' p-composer__button p-composer__button--composer_ia p-composer__selection_button p-composer__button--sticky';
    } else if (context === 'texty_buttons') {
      button.className += ' c-wysiwyg_container__button';
    }
    
    button.setAttribute('role', 'button');
    button.setAttribute('tabindex', '0');
    button.setAttribute('aria-label', 'Robot assistant');
    button.setAttribute('data-qa', 'robot-button');
    button.setAttribute('type', 'button');
    button.setAttribute('data-robot-context', context);
    button.setAttribute('aria-disabled', 'false');
    
    const icon = createRobotIcon();
    button.appendChild(icon);
    
    return button;
  }

  function createLLMSettingsButton() {
    const button = document.createElement('button');
    button.className = 'c-button-unstyled c-icon_button c-icon_button--size_small c-icon_button--default c-wysiwyg_container__button';
    button.setAttribute('role', 'button');
    button.setAttribute('tabindex', '0');
    button.setAttribute('aria-label', 'LLM Settings');
    button.setAttribute('data-qa', 'llm-settings-button');
    button.setAttribute('type', 'button');
    
    const wrapper = document.createElement('span');
    wrapper.style.display = 'inline-flex';
    wrapper.style.alignItems = 'center';
    
    const icon = createSettingsIcon();
    wrapper.appendChild(icon);
    
    const statusIndicator = createStatusIndicator(providerStatus);
    statusIndicator.className = 'robot-llm-status';
    wrapper.appendChild(statusIndicator);
    
    button.appendChild(wrapper);
    
    return button;
  }

  function createLLMModal() {
    // Remove existing modal if any
    const existing = document.getElementById('llm-settings-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'llm-settings-modal';
    modal.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0,0,0,0.5);
      z-index: 9999;
      display: flex;
      align-items: center;
      justify-content: center;
    `;

    const content = document.createElement('div');
    content.style.cssText = `
      background: white;
      border-radius: 8px;
      padding: 24px;
      max-width: 600px;
      width: 90%;
      max-height: 80vh;
      overflow-y: auto;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    `;

    content.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h2 style="margin: 0; font-size: 20px; font-weight: 600;">LLM Provider Settings</h2>
        <button id="llm-modal-close" style="background: none; border: none; font-size: 24px; cursor: pointer; padding: 0; width: 30px; height: 30px;">&times;</button>
      </div>
      
      <div id="llm-current-status" style="margin-bottom: 20px; padding: 12px; background: #f5f5f5; border-radius: 4px;">
        <div style="font-weight: 500; margin-bottom: 4px;">Current Provider</div>
        <div id="llm-current-info">Loading...</div>
      </div>

      <div id="llm-providers-list" style="margin-bottom: 20px;">
        <h3 style="font-size: 16px; margin-bottom: 12px;">Available Providers</h3>
        <div id="llm-providers-container">Loading...</div>
      </div>

      <div id="llm-configure-section" style="border-top: 1px solid #ddd; padding-top: 20px;">
        <h3 style="font-size: 16px; margin-bottom: 12px;">Configure Provider</h3>
        <form id="llm-config-form">
          <div style="margin-bottom: 12px;">
            <label style="display: block; margin-bottom: 4px; font-weight: 500;">Provider</label>
            <select id="llm-provider-select" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
              <option value="anthropic">Anthropic</option>
              <option value="openai">OpenAI</option>
              <option value="ollama">Ollama (Local)</option>
              <option value="openrouter">OpenRouter</option>
            </select>
          </div>
          
          <div style="margin-bottom: 12px;">
            <label style="display: block; margin-bottom: 4px; font-weight: 500;">Model</label>
            <input type="text" id="llm-model-input" placeholder="e.g., gpt-3.5-turbo" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
          </div>
          
          <div id="llm-api-key-group" style="margin-bottom: 12px;">
            <label style="display: block; margin-bottom: 4px; font-weight: 500;">API Key</label>
            <input type="password" id="llm-api-key-input" placeholder="sk-..." style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
          </div>
          
          <div id="llm-base-url-group" style="margin-bottom: 12px;">
            <label style="display: block; margin-bottom: 4px; font-weight: 500;">Base URL (Optional)</label>
            <input type="text" id="llm-base-url-input" placeholder="http://localhost:11434" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
          </div>
          
          <div style="display: flex; gap: 8px; margin-top: 16px;">
            <button type="submit" style="flex: 1; padding: 8px 16px; background: #007a5a; color: white; border: none; border-radius: 4px; font-weight: 500; cursor: pointer;">Configure</button>
            <button type="button" id="llm-test-btn" style="padding: 8px 16px; background: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; font-weight: 500; cursor: pointer;">Test</button>
          </div>
        </form>
      </div>

      <div id="llm-message" style="margin-top: 16px; padding: 12px; border-radius: 4px; display: none;"></div>
    `;

    modal.appendChild(content);
    document.body.appendChild(modal);

    // Load current provider info
    loadProviderInfo();

    // Event listeners
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.remove();
    });

    document.getElementById('llm-modal-close').addEventListener('click', () => modal.remove());
    
    document.getElementById('llm-provider-select').addEventListener('change', (e) => {
      const provider = e.target.value;
      const apiKeyGroup = document.getElementById('llm-api-key-group');
      const baseUrlGroup = document.getElementById('llm-base-url-group');
      
      // Show/hide fields based on provider
      if (provider === 'ollama') {
        apiKeyGroup.style.display = 'none';
        baseUrlGroup.style.display = 'block';
        document.getElementById('llm-base-url-input').value = 'http://localhost:11434';
      } else {
        apiKeyGroup.style.display = 'block';
        baseUrlGroup.style.display = provider === 'anthropic' || provider === 'openai' ? 'none' : 'block';
      }
    });

    document.getElementById('llm-config-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      await configureProvider();
    });

    document.getElementById('llm-test-btn').addEventListener('click', async () => {
      await testProvider();
    });
  }

  async function loadProviderInfo() {
    chrome.runtime.sendMessage({ action: 'getProviderInfo' }, (response) => {
      if (response && response.success) {
        const currentInfo = document.getElementById('llm-current-info');
        const current = response.current;
        
        if (current.active) {
          currentInfo.innerHTML = `
            <div style="color: #0f0;">● ${current.provider} - ${current.model}</div>
            <div style="font-size: 12px; color: #666;">Activated: ${new Date(current.activated_at).toLocaleString()}</div>
          `;
        } else {
          currentInfo.innerHTML = '<div style="color: #f00;">● No active provider</div>';
        }

        // Load providers list
        const container = document.getElementById('llm-providers-container');
        if (response.providers.length > 0) {
          container.innerHTML = response.providers.map(p => `
            <div style="padding: 8px; margin-bottom: 8px; background: #f8f8f8; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
              <div>
                <strong>${p.name}</strong> - ${p.model}
                ${p.is_active ? '<span style="color: #0f0; font-size: 12px;"> (Active)</span>' : ''}
              </div>
              <button onclick="activateProvider('${p.name}')" style="padding: 4px 12px; background: #007a5a; color: white; border: none; border-radius: 4px; font-size: 12px; cursor: pointer;">
                Activate
              </button>
            </div>
          `).join('');
        } else {
          container.innerHTML = '<div style="color: #666;">No providers configured yet</div>';
        }
      }
    });
  }

  async function configureProvider() {
    const provider = document.getElementById('llm-provider-select').value;
    const model = document.getElementById('llm-model-input').value;
    const apiKey = document.getElementById('llm-api-key-input').value;
    const baseUrl = document.getElementById('llm-base-url-input').value;

    if (!model) {
      showMessage('Please enter a model name', 'error');
      return;
    }

    const config = { provider, model };
    if (apiKey) config.api_key = apiKey;
    if (baseUrl) config.base_url = baseUrl;

    chrome.runtime.sendMessage({ action: 'configureProvider', config }, (response) => {
      if (response && response.success) {
        showMessage('Provider configured successfully!', 'success');
        loadProviderInfo();
      } else {
        showMessage(response?.error || 'Failed to configure provider', 'error');
      }
    });
  }

  async function testProvider() {
    showMessage('Testing provider...', 'info');
    
    chrome.runtime.sendMessage({ action: 'testProvider' }, (response) => {
      if (response && response.success) {
        showMessage(`Test successful! Response: "${response.response}"`, 'success');
      } else {
        showMessage(response?.error || 'Provider test failed', 'error');
      }
    });
  }

  function showMessage(text, type) {
    const messageEl = document.getElementById('llm-message');
    messageEl.textContent = text;
    messageEl.style.display = 'block';
    messageEl.style.background = type === 'error' ? '#fee' : type === 'success' ? '#efe' : '#eef';
    messageEl.style.color = type === 'error' ? '#c00' : type === 'success' ? '#080' : '#008';
  }

  // Make activateProvider global for onclick
  window.activateProvider = function(providerName) {
    chrome.runtime.sendMessage({ action: 'activateProvider', provider: providerName }, (response) => {
      if (response && response.success) {
        showMessage(`Provider ${providerName} activated!`, 'success');
        loadProviderInfo();
        fetchCurrentProvider(); // Update status
      } else {
        showMessage(response?.error || 'Failed to activate provider', 'error');
      }
    });
  };

  // Enhanced message data extraction
  function getEnhancedMessageData() {
    // Get conversation ID from URL
    const urlMatch = window.location.pathname.match(/\/client\/([^/]+)\/([^/]+)/);
    const workspaceId = urlMatch ? urlMatch[1] : null;
    const conversationId = urlMatch ? urlMatch[2] : null;

    // Get last message with enhanced context
    const lastMessage = getLastMessageContent();
    
    // Check if we're in a thread
    let threadTs = null;
    const threadContainer = document.querySelector('[data-qa="thread_container"]');
    if (threadContainer) {
      // Try to extract thread timestamp from thread parent
      const threadParent = threadContainer.querySelector('[data-qa="thread_message_root"]');
      if (threadParent && threadParent.id) {
        const tsMatch = threadParent.id.match(/(\d+\.\d+)/);
        if (tsMatch) threadTs = tsMatch[1];
      }
    }

    return {
      user: lastMessage?.user || 'Unknown User',
      content: lastMessage?.content || '',
      timestamp: lastMessage?.timestamp || new Date().toISOString(),
      conversation_id: conversationId,
      thread_ts: threadTs,
      workspace_id: workspaceId
    };
  }

  // Original functions (updated)
  function findInputField() {
    const selectors = [
      'div.ql-editor[contenteditable="true"]',
      'div[data-qa="message_input"] .ql-editor',
      '.c-texty_input_unstyled .ql-editor'
    ];
    
    for (const selector of selectors) {
      const input = document.querySelector(selector);
      if (input) return input;
    }
    return null;
  }

  function insertTextIntoInput(text) {
    const input = findInputField();
    if (!input) {
      console.warn('Could not find Slack input field');
      return;
    }

    input.focus();
    
    // Clear existing content first
    input.innerHTML = '<p><br></p>';
    
    const selection = window.getSelection();
    const range = document.createRange();
    
    const paragraph = input.querySelector('p');
    
    // Clear the paragraph and add new text
    paragraph.innerHTML = '';
    const textNode = document.createTextNode(text);
    paragraph.appendChild(textNode);
    
    // Set cursor at the end
    range.selectNodeContents(paragraph);
    range.collapse(false);
    selection.removeAllRanges();
    selection.addRange(range);
    
    // Remove blank class
    input.classList.remove('ql-blank');
    
    // Trigger input events
    const inputEvent = new Event('input', { bubbles: true, cancelable: true });
    input.dispatchEvent(inputEvent);
    
    const changeEvent = new Event('change', { bubbles: true });
    input.dispatchEvent(changeEvent);
  }

  async function fetchSuggestion(messageData) {
    return new Promise((resolve) => {
      console.log('Sending enhanced message data:', messageData);
      
      chrome.runtime.sendMessage(
        { 
          action: 'fetchSuggestion',
          messageData: messageData 
        },
        (response) => {
          if (chrome.runtime.lastError) {
            console.error('Chrome runtime error:', chrome.runtime.lastError);
            resolve('Failed to fetch suggestion: Extension error');
            return;
          }
          
          if (response && response.success) {
            console.log('Successfully received suggestion:', response.message);
            console.log('Suggestion metadata:', response.metadata);
            resolve(response.message);
          } else {
            console.error('Failed to fetch suggestion:', response);
            resolve(response?.message || 'Failed to fetch suggestion. Is the API server running?');
          }
        }
      );
    });
  }

  function getLastMessageContent() {
    // Find all message elements in the current view
    const messageSelectors = [
      '[id^="message-list_"][role="listitem"]',
      '.c-message_kit__background',
      '[data-qa="message_container"]'
    ];
    
    let messages = [];
    for (const selector of messageSelectors) {
      const found = document.querySelectorAll(selector);
      if (found.length > 0) {
        messages = found;
        break;
      }
    }
    
    if (messages.length === 0) {
      console.log('No messages found in current view');
      return null;
    }
    
    // Get the last message
    const lastMessage = messages[messages.length - 1];
    console.log('Found last message element:', lastMessage);
    
    // Extract message content
    const contentSelectors = [
      '.c-message__message_blocks',
      '.c-message_kit__blocks',
      '[data-qa="message_content"]',
      '.c-message__body'
    ];
    
    let messageContent = null;
    for (const selector of contentSelectors) {
      const content = lastMessage.querySelector(selector);
      if (content) {
        messageContent = content.textContent.trim();
        break;
      }
    }
    
    // Also check for user info
    const userSelectors = [
      '[data-message-sender]',
      '.c-message__sender_button',
      'button[data-qa="message_sender_name"]',
      '.c-message_kit__sender'
    ];
    
    let userName = 'Unknown User';
    for (const selector of userSelectors) {
      const userElement = lastMessage.querySelector(selector);
      if (userElement) {
        userName = userElement.textContent.trim() || userElement.getAttribute('aria-label') || 'Unknown User';
        break;
      }
    }
    
    // Extract timestamp if available
    let timestamp = null;
    const idMatch = lastMessage.id?.match(/message-list_(\d+\.\d+)/);
    if (idMatch) {
      timestamp = idMatch[1];
    }
    
    return {
      content: messageContent,
      user: userName,
      timestamp: timestamp,
      element: lastMessage
    };
  }

  async function handleRobotClick(event) {
    console.log('Robot button clicked!', event);
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    
    const button = event.currentTarget;
    const context = button.getAttribute('data-robot-context');
    
    // Get enhanced message data
    const messageData = getEnhancedMessageData();
    console.log('Enhanced message data:', messageData);
    
    // Add loading state
    button.style.opacity = '0.5';
    button.disabled = true;
    
    let message;
    if (context === 'texty_buttons') {
      // Fetch suggestion from API
      console.log('Fetching enhanced suggestion from API...');
      insertTextIntoInput('Fetching suggestion...');
      
      message = await fetchSuggestion(messageData);
    } else {
      // Use default message for sticky_bar
      message = `Hello World ${context}`;
    }
    
    console.log('Inserting message:', message);
    insertTextIntoInput(message);
    
    // Remove loading state
    button.style.opacity = '1';
    button.disabled = false;
    
    // For sticky bar, we might need to clear the selection
    if (context === 'sticky_bar') {
      const selection = window.getSelection();
      if (selection.rangeCount > 0) {
        selection.removeAllRanges();
      }
    }
  }

  async function handleLLMSettingsClick(event) {
    console.log('LLM Settings button clicked!');
    event.preventDefault();
    event.stopPropagation();
    
    createLLMModal();
  }

  function injectRobotButton() {
    // Inject into sticky bar
    const stickyBar = document.querySelector('.p-texty_sticky_formatting_bar');
    if (stickyBar && !stickyBar.querySelector('[data-qa="robot-button"]')) {
      console.log('Found sticky bar');
      
      const buttonContainer = stickyBar.querySelector('.p-composer__body') || 
                             stickyBar.querySelector('div[style*="overflow"]')?.querySelector('.p-composer__body');
      
      if (buttonContainer) {
        console.log('Found button container');
        
        const referenceButton = buttonContainer.querySelector('button[data-qa="code-block-composer-button"]') ||
                               buttonContainer.querySelector('button[data-qa="code-composer-button"]') ||
                               Array.from(buttonContainer.querySelectorAll('button')).pop();
        
        if (referenceButton) {
          console.log('Found reference button');
          const robotButton = createRobotButton('sticky_bar');
          robotButton.addEventListener('click', handleRobotClick);
          referenceButton.parentNode.insertBefore(robotButton, referenceButton.nextSibling);
          console.log('Robot button added to sticky bar');
        }
      }
    }
    
    // Inject into texty buttons area
    const textyButtonsContainers = document.querySelectorAll('[data-qa="texty_composer_buttons"]');
    textyButtonsContainers.forEach(container => {
      if (!container.querySelector('[data-qa="robot-button"]')) {
        console.log('Found texty buttons container');
        
        const buttonRow = container.querySelector('.c-wysiwyg_container__formatting_buttons') ||
                         container.querySelector('.c-texty_buttons__container') ||
                         container.querySelector('div[role="group"]');
        
        if (buttonRow) {
          console.log('Found button row');
          
          // Add robot button
          const robotButton = createRobotButton('texty_buttons');
          robotButton.addEventListener('click', handleRobotClick);
          
          const referenceButton = buttonRow.querySelector('button[data-qa="texty_code_button"]') ||
                                 buttonRow.querySelector('button[data-qa="texty_code_block_button"]') ||
                                 Array.from(buttonRow.querySelectorAll('button')).pop();
          
          if (referenceButton) {
            referenceButton.parentNode.insertBefore(robotButton, referenceButton.nextSibling);
            console.log('Robot button added to texty buttons');
          } else {
            buttonRow.appendChild(robotButton);
            console.log('Robot button appended to texty buttons');
          }

          // Add LLM settings button
          const settingsButton = createLLMSettingsButton();
          settingsButton.addEventListener('click', handleLLMSettingsClick);
          buttonRow.appendChild(settingsButton);
          console.log('LLM Settings button added');
        }
      }
    });
  }

  function setupObserver() {
    console.log('Setting up mutation observer...');
    observer = new MutationObserver((mutations) => {
      let shouldInject = false;
      
      for (const mutation of mutations) {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
          for (const node of mutation.addedNodes) {
            if (node.nodeType === Node.ELEMENT_NODE) {
              if (node.matches && (
                node.matches('.p-texty_sticky_formatting_bar') ||
                node.matches('[data-qa="texty_composer_buttons"]') ||
                node.querySelector && (
                  node.querySelector('.p-texty_sticky_formatting_bar') ||
                  node.querySelector('[data-qa="texty_composer_buttons"]')
                )
              )) {
                shouldInject = true;
                break;
              }
            }
          }
        }
      }
      
      if (shouldInject) {
        console.log('New message area detected, injecting button...');
        setTimeout(injectRobotButton, 100);
      }
    });
    
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  function initialize() {
    console.log('Slack Robot Assistant initializing...');
    
    // Initial fetch of provider status
    fetchCurrentProvider();
    
    // Initial injection
    injectRobotButton();
    
    // Setup observer
    setupObserver();
    
    // Also check periodically as fallback
    checkInterval = setInterval(() => {
      injectRobotButton();
    }, 2000);
    
    // Refresh provider status periodically
    setInterval(() => {
      fetchCurrentProvider();
    }, 30000); // Every 30 seconds
    
    console.log('Slack Robot Assistant initialized!');
  }

  // Wait for page to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
  } else {
    setTimeout(initialize, 500);
  }
  
  // Re-initialize on navigation
  let lastUrl = location.href;
  new MutationObserver(() => {
    const url = location.href;
    if (url !== lastUrl) {
      lastUrl = url;
      console.log('URL changed, re-initializing...');
      setTimeout(initialize, 1000);
    }
  }).observe(document, { subtree: true, childList: true });
})();