(function() {
  'use strict';

  let observer = null;
  let checkInterval = null;
  let currentProvider = null;
  let providerStatus = 'unknown';

  // Icons using Lucide
  function createRobotIcon() {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', '20');
    svg.setAttribute('height', '20');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    // Using Bot icon from Lucide
    svg.innerHTML = `
      <path d="M12 8V4H8"/>
      <rect width="16" height="12" x="4" y="8" rx="2"/>
      <path d="M2 14h2"/>
      <path d="M20 14h2"/>
      <path d="M15 13v2"/>
      <path d="M9 13v2"/>
    `;
    return svg;
  }

  function createSettingsIcon() {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', '20');
    svg.setAttribute('height', '20');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    // Using simpler Settings-2 icon from Lucide for better rendering at small sizes
    svg.innerHTML = `
      <path d="M20 7h-9"/>
      <path d="M14 17H5"/>
      <circle cx="17" cy="17" r="3"/>
      <circle cx="7" cy="7" r="3"/>
    `;
    return svg;
  }

  function createStatusIndicator(status) {
    const indicator = document.createElement('span');
    indicator.style.cssText = `
      display: inline-block;
      width: 6px;
      height: 6px;
      border-radius: 50%;
      border: 1.5px solid white;
      box-shadow: 0 0 2px rgba(0,0,0,0.3);
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
  function createRobotButton() {
    const button = document.createElement('button');
    button.className = 'c-button-unstyled c-icon_button c-icon_button--size_small c-icon_button--default c-wysiwyg_container__button';
    
    button.setAttribute('role', 'button');
    button.setAttribute('tabindex', '0');
    button.setAttribute('aria-label', 'Robot assistant');
    button.setAttribute('data-qa', 'robot-button');
    button.setAttribute('type', 'button');
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
    wrapper.style.cssText = `
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      height: 100%;
      position: relative;
    `;
    
    const icon = createSettingsIcon();
    icon.style.cssText = `
      display: block;
      width: 20px;
      height: 20px;
    `;
    wrapper.appendChild(icon);
    
    const statusIndicator = createStatusIndicator(providerStatus);
    statusIndicator.className = 'robot-llm-status';
    statusIndicator.style.position = 'absolute';
    statusIndicator.style.bottom = '0px';
    statusIndicator.style.right = '0px';
    wrapper.appendChild(statusIndicator);
    
    button.appendChild(wrapper);
    
    return button;
  }

  function createLLMModal() {
    // Remove existing modal if any
    const existing = document.getElementById('llm-settings-modal');
    if (existing) existing.remove();

    // Debug: Log all CSS variables
    const computedStyle = getComputedStyle(document.documentElement);
    console.log('=== Slack CSS Variables Debug ===');
    
    // Try multiple methods to get the background color
    let primaryBg = '#ffffff';
    let primaryFg = 'rgba(29, 28, 29, 1)';
    let isDarkTheme = false;
    
    // Method 1: Check body classes
    if (document.body.classList.contains('theme-dark') || 
        document.body.classList.contains('theme_dark') ||
        document.body.getAttribute('data-theme') === 'dark') {
      isDarkTheme = true;
    }
    
    // Method 2: Try to get CSS variables from different elements
    const possibleRoots = [document.documentElement, document.body, document.querySelector('.p-client') || document.body];
    for (const root of possibleRoots) {
      const style = getComputedStyle(root);
      const bg = style.getPropertyValue('--sk_primary_background') || 
                 style.getPropertyValue('--sk_primary_background_color') ||
                 style.getPropertyValue('--primary-background') ||
                 style.backgroundColor;
      if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
        primaryBg = bg;
        console.log(`Found background from ${root.tagName}: ${bg}`);
        break;
      }
    }
    
    // Method 3: Check Slack's workspace element
    const workspace = document.querySelector('.p-workspace__primary_view_body');
    if (workspace) {
      const workspaceBg = getComputedStyle(workspace).backgroundColor;
      if (workspaceBg && workspaceBg !== 'rgba(0, 0, 0, 0)') {
        primaryBg = workspaceBg;
        console.log('Found background from workspace:', workspaceBg);
      }
    }
    
    // If we got a dark color, override to dark theme colors
    const bgRgb = primaryBg.match(/\d+/g);
    if (bgRgb) {
      const brightness = (parseInt(bgRgb[0]) + parseInt(bgRgb[1]) + parseInt(bgRgb[2])) / 3;
      if (brightness < 128) {
        isDarkTheme = true;
      }
    }
    
    // Set colors based on theme detection
    if (isDarkTheme) {
      primaryBg = '#1a1d21';  // Slack's dark theme background
      primaryFg = 'rgba(209, 210, 211, 0.9)';  // Slack's dark theme text
    } else {
      primaryBg = '#ffffff';
      primaryFg = 'rgba(29, 28, 29, 0.9)';
    }
    
    const borderColor = isDarkTheme ? 'rgba(209, 210, 211, 0.13)' : 'rgba(29, 28, 29, 0.13)';
    
    console.log('Theme detection result:', { isDarkTheme, primaryBg, primaryFg });

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
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
    `;

    const content = document.createElement('div');
    content.style.cssText = `
      background: ${primaryBg};
      border-radius: 8px;
      padding: 24px;
      max-width: 600px;
      width: 90%;
      max-height: 80vh;
      overflow-y: auto;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
      color: ${primaryFg};
    `;

    // Set additional color values based on theme
    const foregroundLow = isDarkTheme ? 'rgba(209, 210, 211, 0.7)' : 'rgba(29, 28, 29, 0.7)';
    const foregroundHigh = primaryFg;
    const foregroundMax = isDarkTheme ? 'rgba(209, 210, 211, 1)' : 'rgba(29, 28, 29, 1)';
    const highlightBg = '#007a5a';  // Slack's teal color
    const inputBg = isDarkTheme ? 'rgba(255, 255, 255, 0.08)' : '#ffffff';
    
    content.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h2 style="margin: 0; font-size: 18px; font-weight: 700; color: ${foregroundMax};">LLM Provider Settings</h2>
        <button id="llm-modal-close" style="background: none; border: none; font-size: 24px; cursor: pointer; padding: 0; width: 30px; height: 30px; color: ${foregroundHigh}; line-height: 1;">&times;</button>
      </div>
      
      <div id="llm-current-status" style="margin-bottom: 20px; padding: 12px; background: ${isDarkTheme ? 'rgba(0, 0, 0, 0.2)' : 'rgba(0, 0, 0, 0.04)'}; border-radius: 4px; border: 1px solid ${borderColor};">
        <div style="font-weight: 700; margin-bottom: 4px; font-size: 13px; color: ${foregroundMax};">Current Provider</div>
        <div id="llm-current-info" style="font-size: 13px; color: ${foregroundHigh};">Loading...</div>
      </div>

      <div id="llm-providers-list" style="margin-bottom: 20px;">
        <h3 style="font-size: 15px; font-weight: 700; margin-bottom: 12px; color: ${foregroundMax};">Available Providers</h3>
        <div id="llm-providers-container" style="color: ${foregroundHigh};">Loading...</div>
      </div>

      <div id="llm-configure-section" style="border-top: 1px solid ${borderColor}; padding-top: 20px;">
        <h3 style="font-size: 15px; font-weight: 700; margin-bottom: 12px; color: ${foregroundMax};">Configure Provider</h3>
        <form id="llm-config-form">
          <div style="margin-bottom: 12px;">
            <label style="display: block; margin-bottom: 4px; font-weight: 700; font-size: 13px; color: ${foregroundMax};">Provider</label>
            <select id="llm-provider-select" style="width: 100%; padding: 8px; border: 1px solid ${borderColor}; border-radius: 4px; background: ${inputBg}; color: ${foregroundHigh}; font-size: 15px;">
              <option value="anthropic" style="background: ${primaryBg}; color: ${foregroundHigh};">Anthropic</option>
              <option value="openai" style="background: ${primaryBg}; color: ${foregroundHigh};">OpenAI</option>
              <option value="ollama" style="background: ${primaryBg}; color: ${foregroundHigh};">Ollama (Local)</option>
              <option value="openrouter" style="background: ${primaryBg}; color: ${foregroundHigh};">OpenRouter</option>
            </select>
          </div>
          
          <div style="margin-bottom: 12px;">
            <label style="display: block; margin-bottom: 4px; font-weight: 700; font-size: 13px; color: ${foregroundMax};">Model</label>
            <input type="text" id="llm-model-input" placeholder="e.g., gpt-3.5-turbo" style="width: 100%; padding: 8px; border: 1px solid ${borderColor}; border-radius: 4px; background: ${inputBg}; color: ${foregroundHigh}; font-size: 15px;">
            <div id="llm-model-hint" style="font-size: 12px; color: ${foregroundLow}; margin-top: 4px;"></div>
          </div>
          
          <div id="llm-api-key-group" style="margin-bottom: 12px;">
            <label style="display: block; margin-bottom: 4px; font-weight: 700; font-size: 13px; color: ${foregroundMax};">API Key</label>
            <input type="password" id="llm-api-key-input" placeholder="sk-..." style="width: 100%; padding: 8px; border: 1px solid ${borderColor}; border-radius: 4px; background: ${inputBg}; color: ${foregroundHigh}; font-size: 15px;">
          </div>
          
          <div id="llm-base-url-group" style="margin-bottom: 12px;">
            <label style="display: block; margin-bottom: 4px; font-weight: 700; font-size: 13px; color: ${foregroundMax};">Base URL (Optional)</label>
            <input type="text" id="llm-base-url-input" placeholder="http://localhost:11434" style="width: 100%; padding: 8px; border: 1px solid ${borderColor}; border-radius: 4px; background: ${inputBg}; color: ${foregroundHigh}; font-size: 15px;">
          </div>
          
          <div style="display: flex; gap: 8px; margin-top: 16px;">
            <button type="submit" style="flex: 1; padding: 8px 16px; background: ${highlightBg}; color: white; border: none; border-radius: 4px; font-weight: 700; font-size: 15px; cursor: pointer; transition: background 0.1s ease;">Configure</button>
            <button type="button" id="llm-test-btn" style="padding: 8px 16px; background: ${isDarkTheme ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.04)'}; border: 1px solid ${borderColor}; border-radius: 4px; font-weight: 700; font-size: 15px; cursor: pointer; color: ${foregroundHigh}; transition: background 0.1s ease;">Test</button>
          </div>
        </form>
      </div>

      <div id="llm-message" style="margin-top: 16px; padding: 12px; border-radius: 4px; display: none; font-size: 13px;"></div>
    `;

    modal.appendChild(content);
    document.body.appendChild(modal);
    
    // Add styles for select dropdowns in dark theme
    if (isDarkTheme) {
      const style = document.createElement('style');
      style.textContent = `
        #llm-settings-modal select {
          background-color: ${primaryBg} !important;
          color: ${foregroundHigh} !important;
        }
        #llm-settings-modal select option {
          background-color: ${primaryBg} !important;
          color: ${foregroundHigh} !important;
        }
        #llm-settings-modal select:focus {
          outline-color: ${highlightBg};
        }
      `;
      modal.appendChild(style);
    }

    // Load current provider info
    loadProviderInfo();

    // Add hover effects for buttons
    const configureBtn = content.querySelector('button[type="submit"]');
    const testBtn = document.getElementById('llm-test-btn');
    
    configureBtn.addEventListener('mouseenter', () => {
      configureBtn.style.opacity = '0.8';
    });
    configureBtn.addEventListener('mouseleave', () => {
      configureBtn.style.opacity = '1';
    });
    
    testBtn.addEventListener('mouseenter', () => {
      testBtn.style.background = isDarkTheme ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 0, 0, 0.08)';
    });
    testBtn.addEventListener('mouseleave', () => {
      testBtn.style.background = isDarkTheme ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.04)';
    });

    // Event listeners
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.remove();
    });

    const closeBtn = document.getElementById('llm-modal-close');
    closeBtn.addEventListener('click', () => modal.remove());
    closeBtn.addEventListener('mouseenter', () => {
      closeBtn.style.opacity = '1';
      closeBtn.style.background = isDarkTheme ? 'rgba(0, 0, 0, 0.3)' : 'rgba(29, 28, 29, 0.08)';
      closeBtn.style.borderRadius = '4px';
    });
    closeBtn.addEventListener('mouseleave', () => {
      closeBtn.style.opacity = '';
      closeBtn.style.background = 'none';
    });
    
    // Trigger provider change to show initial hints
    document.getElementById('llm-provider-select').dispatchEvent(new Event('change'));
    
    document.getElementById('llm-provider-select').addEventListener('change', (e) => {
      const provider = e.target.value;
      const apiKeyGroup = document.getElementById('llm-api-key-group');
      const baseUrlGroup = document.getElementById('llm-base-url-group');
      const modelHint = document.getElementById('llm-model-hint');
      const modelInput = document.getElementById('llm-model-input');
      
      // Show/hide fields and set hints based on provider
      if (provider === 'ollama') {
        apiKeyGroup.style.display = 'none';
        baseUrlGroup.style.display = 'block';
        document.getElementById('llm-base-url-input').value = 'http://localhost:11434';
        modelHint.textContent = 'Examples: llama2, codellama, mistral, neural-chat';
        modelInput.placeholder = 'e.g., llama2';
      } else if (provider === 'anthropic') {
        apiKeyGroup.style.display = 'block';
        baseUrlGroup.style.display = 'none';
        document.getElementById('llm-base-url-input').value = ''; // Clear base URL
        modelHint.textContent = 'Examples: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307';
        modelInput.placeholder = 'e.g., claude-3-sonnet-20240229';
      } else if (provider === 'openai') {
        apiKeyGroup.style.display = 'block';
        baseUrlGroup.style.display = 'none';
        document.getElementById('llm-base-url-input').value = ''; // Clear base URL
        modelHint.textContent = 'Examples: gpt-4-turbo-preview, gpt-4, gpt-3.5-turbo';
        modelInput.placeholder = 'e.g., gpt-3.5-turbo';
      } else if (provider === 'openrouter') {
        apiKeyGroup.style.display = 'block';
        baseUrlGroup.style.display = 'block';
        modelHint.textContent = 'Examples: openai/gpt-3.5-turbo, anthropic/claude-2, meta-llama/llama-2-70b-chat';
        modelInput.placeholder = 'e.g., openai/gpt-3.5-turbo';
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
        // Get computed styles from the modal if it exists, or detect theme
        const modal = document.getElementById('llm-settings-modal');
        let isDarkTheme = false;
        let foregroundHigh, foregroundLow, foregroundMax;
        
        // First try to detect from body classes
        if (document.body.classList.contains('theme-dark') || 
            document.body.classList.contains('theme_dark') ||
            document.body.getAttribute('data-theme') === 'dark') {
          isDarkTheme = true;
        }
        
        // Try to get actual colors from Slack's workspace
        const workspace = document.querySelector('.p-workspace__primary_view_body');
        if (workspace) {
          const workspaceStyle = getComputedStyle(workspace);
          const bgColor = workspaceStyle.backgroundColor;
          const textColor = workspaceStyle.color;
          
          // If we got valid colors, use them
          if (textColor && textColor !== 'rgba(0, 0, 0, 0)') {
            foregroundHigh = textColor;
            foregroundMax = textColor;
            foregroundLow = textColor.replace(/[\d.]+\)$/g, '0.7)'); // Make low opacity version
            
            console.log('Using workspace colors - text:', textColor, 'bg:', bgColor);
          }
        }
        
        // Fallback to default colors if we couldn't get them
        if (!foregroundHigh) {
          foregroundHigh = isDarkTheme ? '#d1d2d3' : '#1d1c1d';
          foregroundLow = isDarkTheme ? '#9a9b9c' : '#616061';
          foregroundMax = isDarkTheme ? '#e8e8e8' : '#1d1c1d';
        }
        
        const highlightBg = '#007a5a';
        const borderColor = isDarkTheme ? 'rgba(209, 210, 211, 0.13)' : 'rgba(29, 28, 29, 0.13)';
        
        const currentInfo = document.getElementById('llm-current-info');
        const current = response.current;
        
        if (current.active) {
          currentInfo.innerHTML = `
            <div style="color: ${highlightBg}; font-weight: 700;">● ${current.provider} - ${current.model}</div>
            <div style="font-size: 12px; color: ${foregroundLow};">Activated: ${new Date(current.activated_at).toLocaleString()}</div>
          `;
        } else {
          currentInfo.innerHTML = `<div style="color: ${foregroundLow};">● No active provider</div>`;
        }

        // Load providers list
        const container = document.getElementById('llm-providers-container');
        
        // Debug log final colors being used
        console.log('=== Final Provider List Colors ===');
        console.log('isDarkTheme:', isDarkTheme);
        console.log('foregroundMax:', foregroundMax);
        console.log('foregroundHigh:', foregroundHigh);
        console.log('highlightBg:', highlightBg);
        
        if (response.providers.length > 0) {
          container.innerHTML = response.providers.map(p => `
            <div style="padding: 12px; margin-bottom: 8px; background: ${isDarkTheme ? 'rgba(0, 0, 0, 0.2)' : 'rgba(0, 0, 0, 0.04)'}; border: 1px solid ${borderColor}; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
              <div>
                <strong style="font-size: 13px; color: ${foregroundMax} !important; display: inline;">${p.name}</strong> <span style="color: ${foregroundHigh} !important; font-size: 13px; display: inline;">- ${p.model}</span>
                ${p.is_active ? `<span style="color: ${highlightBg} !important; font-size: 12px; font-weight: 700; display: inline;"> (Active)</span>` : ''}
              </div>
              <button class="activate-provider-btn" data-provider="${p.name}" style="padding: 6px 12px; background: ${highlightBg}; color: white !important; border: none; border-radius: 4px; font-size: 13px; font-weight: 700; cursor: pointer; transition: opacity 0.1s ease;">
                Activate
              </button>
            </div>
          `).join('');
          
          // Add click handlers and hover effects to activate buttons
          container.querySelectorAll('.activate-provider-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
              const providerName = e.target.getAttribute('data-provider');
              activateProvider(providerName);
            });
            
            btn.addEventListener('mouseenter', () => {
              btn.style.opacity = '0.8';
            });
            btn.addEventListener('mouseleave', () => {
              btn.style.opacity = '1';
            });
          });
        } else {
          container.innerHTML = `<div style="color: ${foregroundLow}; font-size: 13px;">No providers configured yet</div>`;
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

    // Validate required fields based on provider
    if (provider !== 'ollama' && !apiKey) {
      showMessage('Please enter an API key', 'error');
      return;
    }

    const config = { provider, model };
    if (apiKey) config.api_key = apiKey;
    
    // Only include base_url if it's needed and not empty
    if (baseUrl && baseUrl.trim() !== '') {
      // Only include base_url for providers that need it
      if (provider === 'ollama' || provider === 'openrouter') {
        config.base_url = baseUrl;
      }
    }

    showMessage('Configuring provider...', 'info');

    chrome.runtime.sendMessage({ action: 'configureProvider', config }, (response) => {
      if (response && response.success) {
        showMessage('Provider configured successfully! You can now activate or test it.', 'success');
        loadProviderInfo();
      } else {
        showMessage(response?.error || 'Failed to configure provider', 'error');
      }
    });
  }

  async function testProvider() {
    // First check if there's an active provider
    chrome.runtime.sendMessage({ action: 'getCurrentProvider' }, (providerResponse) => {
      if (!providerResponse || !providerResponse.success || !providerResponse.data.active) {
        showMessage('No active provider. Please configure and activate a provider first.', 'error');
        return;
      }
      
      showMessage('Testing provider...', 'info');
      
      chrome.runtime.sendMessage({ action: 'testProvider' }, (response) => {
        if (response && response.success) {
          showMessage(`Test successful! Response: "${response.response}"`, 'success');
        } else {
          showMessage(response?.error || 'Provider test failed', 'error');
        }
      });
    });
  }

  function showMessage(text, type) {
    // Get CSS variables
    const computedStyle = getComputedStyle(document.documentElement);
    const highlightBg = computedStyle.getPropertyValue('--sk_highlight') || '#007a5a';
    const errorColor = computedStyle.getPropertyValue('--sk_highlight_urgent') || '#d73a49';
    const infoColor = computedStyle.getPropertyValue('--sk_highlight_hover') || '#1264a3';
    const borderColor = computedStyle.getPropertyValue('--sk_foreground_low') || 'rgba(29, 28, 29, 0.13)';
    
    const messageEl = document.getElementById('llm-message');
    messageEl.textContent = text;
    messageEl.style.display = 'block';
    
    if (type === 'error') {
      messageEl.style.background = 'rgba(235, 87, 87, 0.08)';
      messageEl.style.color = errorColor;
      messageEl.style.border = `1px solid rgba(235, 87, 87, 0.2)`;
    } else if (type === 'success') {
      messageEl.style.background = 'rgba(0, 122, 90, 0.08)';
      messageEl.style.color = highlightBg;
      messageEl.style.border = `1px solid rgba(0, 122, 90, 0.2)`;
    } else {
      messageEl.style.background = 'rgba(29, 155, 209, 0.08)';
      messageEl.style.color = infoColor;
      messageEl.style.border = `1px solid ${borderColor}`;
    }
  }

  function activateProvider(providerName) {
    showMessage(`Activating provider ${providerName}...`, 'info');
    chrome.runtime.sendMessage({ action: 'activateProvider', provider: providerName }, (response) => {
      if (response && response.success) {
        showMessage(`Provider ${providerName} activated successfully!`, 'success');
        loadProviderInfo();
        fetchCurrentProvider(); // Update status
      } else {
        showMessage(response?.error || 'Failed to activate provider', 'error');
      }
    });
  }

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

  function createLucideIcon(iconName) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', '16');
    svg.setAttribute('height', '16');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    svg.style.flexShrink = '0';
    
    const iconPaths = {
      'bot': `<path d="M12 8V4H8"/>
              <rect width="16" height="12" x="4" y="8" rx="2"/>
              <path d="M2 14h2"/>
              <path d="M20 14h2"/>
              <path d="M15 13v2"/>
              <path d="M9 13v2"/>`,
      'book': `<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>`,
      'settings': `<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>
                   <circle cx="12" cy="12" r="3"/>`,
      'help': `<circle cx="12" cy="12" r="10"/>
               <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
               <path d="M12 17h.01"/>`,
      'info': `<circle cx="12" cy="12" r="10"/>
               <path d="M12 16v-4"/>
               <path d="M12 8h.01"/>`
    };
    
    svg.innerHTML = iconPaths[iconName] || '';
    return svg;
  }

  function createRobotContextMenu(x, y) {
    // Remove existing context menu if any
    const existing = document.getElementById('robot-context-menu');
    if (existing) existing.remove();

    // Use the same theme detection logic as the modal
    let primaryBg = '#ffffff';
    let primaryFg = 'rgba(29, 28, 29, 1)';
    let isDarkTheme = false;
    
    // Method 1: Check body classes
    if (document.body.classList.contains('theme-dark') || 
        document.body.classList.contains('theme_dark') ||
        document.body.getAttribute('data-theme') === 'dark') {
      isDarkTheme = true;
    }
    
    // Method 2: Try to get CSS variables from different elements
    const possibleRoots = [document.documentElement, document.body, document.querySelector('.p-client') || document.body];
    for (const root of possibleRoots) {
      const style = getComputedStyle(root);
      const bg = style.getPropertyValue('--sk_primary_background') || 
                 style.getPropertyValue('--sk_primary_background_color') ||
                 style.getPropertyValue('--primary-background') ||
                 style.backgroundColor;
      if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
        primaryBg = bg;
        console.log(`Found context menu background from ${root.tagName}: ${bg}`);
        break;
      }
    }
    
    // Method 3: Check Slack's workspace element
    const workspace = document.querySelector('.p-workspace__primary_view_body');
    if (workspace) {
      const workspaceBg = getComputedStyle(workspace).backgroundColor;
      if (workspaceBg && workspaceBg !== 'rgba(0, 0, 0, 0)') {
        primaryBg = workspaceBg;
        console.log('Found context menu background from workspace:', workspaceBg);
      }
    }
    
    // If we got a dark color, override to dark theme colors
    const bgRgb = primaryBg.match(/\d+/g);
    if (bgRgb) {
      const brightness = (parseInt(bgRgb[0]) + parseInt(bgRgb[1]) + parseInt(bgRgb[2])) / 3;
      if (brightness < 128) {
        isDarkTheme = true;
      }
    }
    
    // Set colors based on theme detection
    if (isDarkTheme) {
      primaryBg = '#1a1d21';  // Slack's dark theme background
      primaryFg = 'rgba(209, 210, 211, 0.9)';  // Slack's dark theme text
    } else {
      primaryBg = '#ffffff';
      primaryFg = 'rgba(29, 28, 29, 0.9)';
    }
    
    const borderColor = isDarkTheme ? 'rgba(209, 210, 211, 0.13)' : 'rgba(29, 28, 29, 0.13)';
    
    console.log('Context menu theme detection result:', { isDarkTheme, primaryBg, primaryFg });

    const menu = document.createElement('div');
    menu.id = 'robot-context-menu';
    menu.style.cssText = `
      position: fixed;
      top: ${y}px;
      left: ${x}px;
      background: ${primaryBg};
      border: 1px solid ${borderColor};
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      z-index: 10000;
      min-width: 240px;
      padding: 8px 0;
      font-family: Slack-Lato, Slack-Fractions, appleLogo, sans-serif;
      font-size: 15px;
      line-height: 1.46668;
    `;

    const menuItems = [
      { label: 'Generate suggestion', icon: 'bot', action: 'generate' },
      { label: 'Add knowledge', icon: 'book', action: 'add-knowledge' },
      { label: 'Settings', icon: 'settings', action: 'settings' },
      { label: 'Help', icon: 'help', action: 'help' },
      { label: 'About', icon: 'info', action: 'about' }
    ];

    menuItems.forEach((item, index) => {
      const menuItem = document.createElement('div');
      menuItem.style.cssText = `
        padding: 8px 16px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 12px;
        color: ${primaryFg};
        font-size: 15px;
        font-weight: 400;
        line-height: 1.46668;
        transition: background-color 0.1s ease-out;
        user-select: none;
      `;

      const icon = createLucideIcon(item.icon);
      const iconColor = isDarkTheme ? 'rgba(209, 210, 211, 0.7)' : 'rgba(29, 28, 29, 0.7)';
      icon.style.color = iconColor;
      
      const textSpan = document.createElement('span');
      textSpan.textContent = item.label;
      textSpan.style.cssText = `
        flex: 1;
        font-weight: 400;
      `;

      menuItem.appendChild(icon);
      menuItem.appendChild(textSpan);

      // Add hover effects - matching Slack's exact hover behavior
      menuItem.addEventListener('mouseenter', () => {
        menuItem.style.backgroundColor = 'rgba(29, 155, 209, 1)';
        menuItem.style.color = '#ffffff';
        icon.style.color = '#ffffff';
      });
      menuItem.addEventListener('mouseleave', () => {
        menuItem.style.backgroundColor = 'transparent';
        menuItem.style.color = primaryFg;
        icon.style.color = iconColor;
      });

      // Add click handler
      menuItem.addEventListener('click', () => {
        handleContextMenuAction(item.action);
        menu.remove();
      });

      menu.appendChild(menuItem);

      // Add separator after Settings - using Slack's exact separator style
      if (index === 2) {
        const separator = document.createElement('div');
        separator.style.cssText = `
          height: 1px;
          background: ${isDarkTheme ? 'rgba(209, 210, 211, 0.13)' : 'rgba(29, 28, 29, 0.13)'};
          margin: 8px 0;
        `;
        menu.appendChild(separator);
      }
    });

    document.body.appendChild(menu);

    // Remove menu when clicking outside
    const removeMenu = (e) => {
      if (!menu.contains(e.target)) {
        menu.remove();
        document.removeEventListener('click', removeMenu);
        document.removeEventListener('keydown', handleKeydown);
      }
    };
    
    // Add keyboard support
    const handleKeydown = (e) => {
      if (e.key === 'Escape') {
        menu.remove();
        document.removeEventListener('click', removeMenu);
        document.removeEventListener('keydown', handleKeydown);
      }
    };
    
    setTimeout(() => {
      document.addEventListener('click', removeMenu);
      document.addEventListener('keydown', handleKeydown);
    }, 10);

    // Adjust position if menu goes off screen
    const rect = menu.getBoundingClientRect();
    if (rect.right > window.innerWidth) {
      menu.style.left = (x - rect.width) + 'px';
    }
    if (rect.bottom > window.innerHeight) {
      menu.style.top = (y - rect.height) + 'px';
    }
  }

  function createAddKnowledgeModal() {
    // Remove existing modal if any
    const existing = document.getElementById('add-knowledge-modal');
    if (existing) existing.remove();

    // Use the EXACT same theme detection logic as the LLM Settings modal
    const computedStyle = getComputedStyle(document.documentElement);
    console.log('=== Add Knowledge Modal - Slack CSS Variables Debug ===');
    
    // Try multiple methods to get the background color
    let primaryBg = '#ffffff';
    let primaryFg = 'rgba(29, 28, 29, 1)';
    let isDarkTheme = false;
    
    // Method 1: Check body classes
    if (document.body.classList.contains('theme-dark') || 
        document.body.classList.contains('theme_dark') ||
        document.body.getAttribute('data-theme') === 'dark') {
      isDarkTheme = true;
    }
    
    // Method 2: Try to get CSS variables from different elements
    const possibleRoots = [document.documentElement, document.body, document.querySelector('.p-client') || document.body];
    for (const root of possibleRoots) {
      const style = getComputedStyle(root);
      const bg = style.getPropertyValue('--sk_primary_background') || 
                 style.getPropertyValue('--sk_primary_background_color') ||
                 style.getPropertyValue('--primary-background') ||
                 style.backgroundColor;
      if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
        primaryBg = bg;
        console.log(`Found background from ${root.tagName}: ${bg}`);
        break;
      }
    }
    
    // Method 3: Check Slack's workspace element
    const workspace = document.querySelector('.p-workspace__primary_view_body');
    if (workspace) {
      const workspaceBg = getComputedStyle(workspace).backgroundColor;
      if (workspaceBg && workspaceBg !== 'rgba(0, 0, 0, 0)') {
        primaryBg = workspaceBg;
        console.log('Found background from workspace:', workspaceBg);
      }
    }
    
    // If we got a dark color, override to dark theme colors
    const bgRgb = primaryBg.match(/\d+/g);
    if (bgRgb) {
      const brightness = (parseInt(bgRgb[0]) + parseInt(bgRgb[1]) + parseInt(bgRgb[2])) / 3;
      if (brightness < 128) {
        isDarkTheme = true;
      }
    }
    
    // Set colors based on theme detection
    if (isDarkTheme) {
      primaryBg = '#1a1d21';  // Slack's dark theme background
      primaryFg = 'rgba(209, 210, 211, 0.9)';  // Slack's dark theme text
    } else {
      primaryBg = '#ffffff';
      primaryFg = 'rgba(29, 28, 29, 0.9)';
    }
    
    console.log('Add Knowledge Modal theme detection result:', { isDarkTheme, primaryBg, primaryFg });
    
    const borderColor = isDarkTheme ? 'rgba(209, 210, 211, 0.13)' : 'rgba(29, 28, 29, 0.13)';
    const foregroundLow = isDarkTheme ? 'rgba(209, 210, 211, 0.7)' : 'rgba(29, 28, 29, 0.7)';
    const foregroundMax = isDarkTheme ? 'rgba(209, 210, 211, 1)' : 'rgba(29, 28, 29, 1)';
    const highlightBg = '#007a5a';  // Slack's teal color
    const inputBg = isDarkTheme ? 'rgba(255, 255, 255, 0.08)' : '#ffffff';

    const modal = document.createElement('div');
    modal.id = 'add-knowledge-modal';
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
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
    `;

    const content = document.createElement('div');
    content.style.cssText = `
      background: ${primaryBg};
      border-radius: 8px;
      padding: 24px;
      max-width: 600px;
      width: 90%;
      max-height: 80vh;
      overflow-y: auto;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
      color: ${primaryFg};
    `;

    content.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h2 style="margin: 0; font-size: 18px; font-weight: 700; color: ${foregroundMax};">Add Knowledge</h2>
        <button id="knowledge-modal-close" style="background: none; border: none; font-size: 24px; cursor: pointer; padding: 0; width: 30px; height: 30px; color: ${primaryFg}; line-height: 1;">&times;</button>
      </div>
      
      <div style="margin-bottom: 24px;">
        <p style="margin: 0 0 16px 0; font-size: 15px; color: ${foregroundLow}; line-height: 1.46668;">
          Upload documents to expand the AI's knowledge base. Supported formats: PDF, TXT, DOCX, MD
        </p>
      </div>

      <div id="drop-zone" style="
        border: 2px dashed ${borderColor};
        border-radius: 8px;
        padding: 48px 24px;
        text-align: center;
        margin-bottom: 24px;
        background: ${isDarkTheme ? 'rgba(0, 0, 0, 0.1)' : 'rgba(0, 0, 0, 0.02)'};
        transition: all 0.2s ease;
        cursor: pointer;
      ">
        <div style="margin-bottom: 16px;">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="${foregroundLow}" stroke-width="1.5" style="margin: 0 auto; display: block;">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7,10 12,15 17,10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
        </div>
        <div style="font-size: 16px; font-weight: 500; color: ${foregroundMax}; margin-bottom: 8px;">
          Drop files here or click to browse
        </div>
        <div style="font-size: 14px; color: ${foregroundLow};">
          Maximum file size: 10MB
        </div>
      </div>

      <input type="file" id="file-input" multiple accept=".pdf,.txt,.docx,.md" style="display: none;">

      <div id="file-list" style="margin-bottom: 24px; display: none;">
        <h3 style="font-size: 15px; font-weight: 700; margin-bottom: 12px; color: ${foregroundMax};">Selected Files</h3>
        <div id="file-items" style="
          max-height: 156px;
          overflow-y: auto;
          padding-right: 4px;
          scrollbar-width: thin;
          scrollbar-color: ${isDarkTheme ? 'rgba(209, 210, 211, 0.3)' : 'rgba(29, 28, 29, 0.3)'} transparent;
        "></div>
      </div>

      <div style="display: flex; gap: 8px; justify-content: flex-end;">
        <button id="knowledge-cancel-btn" style="padding: 8px 16px; background: ${isDarkTheme ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.04)'}; border: 1px solid ${borderColor}; border-radius: 4px; font-weight: 700; font-size: 15px; cursor: pointer; color: ${primaryFg}; transition: background 0.1s ease;">
          Cancel
        </button>
        <button id="knowledge-upload-btn" style="padding: 8px 16px; background: ${highlightBg}; color: white; border: none; border-radius: 4px; font-weight: 700; font-size: 15px; cursor: pointer; transition: background 0.1s ease; opacity: 0.5;" disabled>
          Upload Files
        </button>
      </div>

      <div id="knowledge-progress" style="margin-top: 16px; display: none;">
        <div style="background: ${isDarkTheme ? 'rgba(0, 0, 0, 0.2)' : 'rgba(0, 0, 0, 0.04)'}; border-radius: 4px; height: 8px; overflow: hidden;">
          <div id="progress-bar" style="background: ${highlightBg}; height: 100%; width: 0%; transition: width 0.3s ease;"></div>
        </div>
        <div id="progress-text" style="margin-top: 8px; font-size: 13px; color: ${foregroundLow}; text-align: center;"></div>
      </div>
    `;

    modal.appendChild(content);
    document.body.appendChild(modal);

    // Add custom scrollbar styles for WebKit browsers
    const style = document.createElement('style');
    style.textContent = `
      #file-items::-webkit-scrollbar {
        width: 8px;
      }
      #file-items::-webkit-scrollbar-track {
        background: transparent;
      }
      #file-items::-webkit-scrollbar-thumb {
        background: ${isDarkTheme ? 'rgba(209, 210, 211, 0.3)' : 'rgba(29, 28, 29, 0.3)'};
        border-radius: 4px;
      }
      #file-items::-webkit-scrollbar-thumb:hover {
        background: ${isDarkTheme ? 'rgba(209, 210, 211, 0.5)' : 'rgba(29, 28, 29, 0.5)'};
      }
    `;
    modal.appendChild(style);

    // Setup functionality
    setupAddKnowledgeModal(modal, isDarkTheme, foregroundMax, foregroundLow, borderColor, highlightBg);
  }

  function setupAddKnowledgeModal(modal, isDarkTheme, foregroundMax, foregroundLow, borderColor, highlightBg) {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const fileItems = document.getElementById('file-items');
    const uploadBtn = document.getElementById('knowledge-upload-btn');
    const cancelBtn = document.getElementById('knowledge-cancel-btn');
    const closeBtn = document.getElementById('knowledge-modal-close');
    const progress = document.getElementById('knowledge-progress');
    
    let selectedFiles = [];

    // Close modal events
    const closeModal = () => modal.remove();
    closeBtn.addEventListener('click', closeModal);
    cancelBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeModal();
    });

    // File input click
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag and drop events
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = highlightBg;
      dropZone.style.background = isDarkTheme ? 'rgba(0, 122, 90, 0.1)' : 'rgba(0, 122, 90, 0.05)';
    });

    dropZone.addEventListener('dragleave', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = borderColor;
      dropZone.style.background = isDarkTheme ? 'rgba(0, 0, 0, 0.1)' : 'rgba(0, 0, 0, 0.02)';
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = borderColor;
      dropZone.style.background = isDarkTheme ? 'rgba(0, 0, 0, 0.1)' : 'rgba(0, 0, 0, 0.02)';
      
      const files = Array.from(e.dataTransfer.files);
      handleFiles(files);
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
      const files = Array.from(e.target.files);
      handleFiles(files);
    });

    function handleFiles(files) {
      const validFiles = files.filter(file => {
        const validTypes = ['.pdf', '.txt', '.docx', '.md'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();
        const validSize = file.size <= 10 * 1024 * 1024; // 10MB
        return validTypes.includes(fileExt) && validSize;
      });

      selectedFiles = [...selectedFiles, ...validFiles];
      updateFileList();
    }

    function updateFileList() {
      if (selectedFiles.length === 0) {
        fileList.style.display = 'none';
        uploadBtn.disabled = true;
        uploadBtn.style.opacity = '0.5';
        return;
      }

      fileList.style.display = 'block';
      uploadBtn.disabled = false;
      uploadBtn.style.opacity = '1';

      fileItems.innerHTML = selectedFiles.map((file, index) => `
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: ${isDarkTheme ? 'rgba(0, 0, 0, 0.2)' : 'rgba(0, 0, 0, 0.04)'}; border-radius: 4px; margin-bottom: 8px;">
          <div style="flex: 1; min-width: 0;">
            <div style="font-size: 14px; font-weight: 500; color: ${foregroundMax}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${file.name}</div>
            <div style="font-size: 12px; color: ${foregroundLow};">${(file.size / 1024).toFixed(1)} KB</div>
          </div>
          <button class="remove-file" data-index="${index}" style="background: none; border: none; color: ${foregroundLow}; cursor: pointer; padding: 4px; font-size: 16px;">&times;</button>
        </div>
      `).join('');

      // Add remove file listeners
      fileItems.querySelectorAll('.remove-file').forEach(btn => {
        btn.addEventListener('click', (e) => {
          const index = parseInt(e.target.getAttribute('data-index'));
          selectedFiles.splice(index, 1);
          updateFileList();
        });
      });
    }

    // Upload functionality (real API calls)
    uploadBtn.addEventListener('click', async () => {
      if (selectedFiles.length === 0) return;
      
      progress.style.display = 'block';
      uploadBtn.disabled = true;
      uploadBtn.style.opacity = '0.5';
      
      const progressBar = document.getElementById('progress-bar');
      const progressText = document.getElementById('progress-text');
      
      try {
        progressText.textContent = 'Preparing files...';
        progressBar.style.width = '10%';
        
        // Convert files to format expected by background script
        const fileDataArray = [];
        
        // Validate file formats first
        const supportedFormats = ['.pdf', '.docx', '.pptx', '.html', '.md', '.csv', '.xlsx', '.png', '.jpg', '.jpeg', '.txt'];
        const invalidFiles = [];
        
        for (let i = 0; i < selectedFiles.length; i++) {
          const file = selectedFiles[i];
          const extension = '.' + file.name.split('.').pop().toLowerCase();
          
          if (!supportedFormats.includes(extension)) {
            invalidFiles.push(`${file.name} (${extension})`);
          }
        }
        
        if (invalidFiles.length > 0) {
          throw new Error(`Unsupported file formats: ${invalidFiles.join(', ')}. Supported: ${supportedFormats.join(', ')}`);
        }
        
        // Validate file sizes (Docling has 100MB limit, but we use 50MB for safety)
        const maxSize = 50 * 1024 * 1024; // 50MB
        const oversizedFiles = [];
        
        for (let i = 0; i < selectedFiles.length; i++) {
          const file = selectedFiles[i];
          if (file.size > maxSize) {
            oversizedFiles.push(`${file.name} (${(file.size / 1024 / 1024).toFixed(1)}MB)`);
          }
        }
        
        if (oversizedFiles.length > 0) {
          throw new Error(`Files too large: ${oversizedFiles.join(', ')}. Maximum size: 50MB`);
        }
        
        for (let i = 0; i < selectedFiles.length; i++) {
          const file = selectedFiles[i];
          
          progressText.textContent = `Processing file ${i + 1}/${selectedFiles.length}...`;
          progressBar.style.width = `${10 + (i / selectedFiles.length) * 30}%`;
          
          // Convert file to data URL
          const dataUrl = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
          });
          
          fileDataArray.push({
            name: file.name,
            type: file.type,
            size: file.size,
            dataUrl: dataUrl
          });
        }
        
        progressText.textContent = 'Uploading to server...';
        progressBar.style.width = '50%';
        
        // Call background script to upload files
        const response = await new Promise((resolve) => {
          chrome.runtime.sendMessage({
            action: 'uploadDocuments',
            files: fileDataArray,
            uploadedBy: 'slack_user'
          }, resolve);
        });
        
        progressBar.style.width = '90%';
        
        if (response.success) {
          progressBar.style.width = '100%';
          progressText.textContent = `Successfully uploaded ${response.data.documents_processed} file(s) with ${response.data.total_chunks} chunks`;
          
          // Log detailed results
          console.log('Upload successful:', response.data);
          if (response.data.file_details) {
            response.data.file_details.forEach(file => {
              console.log(`- ${file.filename}: ${file.status} (${file.chunks} chunks)`);
            });
          }
          
          setTimeout(() => {
            closeModal();
          }, 2000);
        } else {
          // Handle different error formats properly
          let errorMessage = 'Upload failed';
          if (response.error) {
            if (typeof response.error === 'string') {
              errorMessage = response.error;
            } else if (response.error.detail) {
              if (typeof response.error.detail === 'string') {
                errorMessage = response.error.detail;
              } else if (Array.isArray(response.error.detail)) {
                // Handle Pydantic validation errors
                const firstError = response.error.detail[0];
                if (firstError && firstError.msg) {
                  errorMessage = firstError.msg;
                } else {
                  errorMessage = 'Validation error occurred';
                }
              } else {
                errorMessage = JSON.stringify(response.error.detail);
              }
            } else if (response.error.message) {
              errorMessage = response.error.message;
            } else {
              // If it's an object, stringify it for debugging
              errorMessage = `Upload failed: ${JSON.stringify(response.error)}`;
            }
          }
          console.error('Upload failed with error:', response.error);
          throw new Error(errorMessage);
        }
        
      } catch (error) {
        console.error('Upload error:', error);
        progressBar.style.width = '0%';
        progressText.textContent = `Upload failed: ${error.message}`;
        progressText.style.color = '#d73a49';
        
        // Re-enable upload button after error
        setTimeout(() => {
          uploadBtn.disabled = false;
          uploadBtn.style.opacity = '1';
          progress.style.display = 'none';
          progressText.style.color = foregroundLow;
        }, 3000);
      }
    });

    // Button hover effects
    uploadBtn.addEventListener('mouseenter', () => {
      if (!uploadBtn.disabled) {
        uploadBtn.style.opacity = '0.8';
      }
    });
    uploadBtn.addEventListener('mouseleave', () => {
      if (!uploadBtn.disabled) {
        uploadBtn.style.opacity = '1';
      }
    });

    cancelBtn.addEventListener('mouseenter', () => {
      cancelBtn.style.background = isDarkTheme ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 0, 0, 0.08)';
    });
    cancelBtn.addEventListener('mouseleave', () => {
      cancelBtn.style.background = isDarkTheme ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.04)';
    });

    closeBtn.addEventListener('mouseenter', () => {
      closeBtn.style.background = isDarkTheme ? 'rgba(0, 0, 0, 0.3)' : 'rgba(29, 28, 29, 0.08)';
      closeBtn.style.borderRadius = '4px';
    });
    closeBtn.addEventListener('mouseleave', () => {
      closeBtn.style.background = 'none';
    });
  }

  function handleContextMenuAction(action) {
    console.log('Context menu action:', action);
    
    switch (action) {
      case 'generate':
        // Trigger the normal robot click functionality
        const robotButton = document.querySelector('[data-qa="robot-button"]');
        if (robotButton) {
          const fakeEvent = { 
            currentTarget: robotButton, 
            preventDefault: () => {}, 
            stopPropagation: () => {}, 
            stopImmediatePropagation: () => {} 
          };
          handleRobotClick(fakeEvent);
        }
        break;
      case 'add-knowledge':
        // Open Add Knowledge modal
        createAddKnowledgeModal();
        break;
      case 'settings':
        // Open LLM settings modal
        createLLMModal();
        break;
      case 'help':
        console.log('Help feature - to be implemented');
        insertTextIntoInput('Help documentation coming soon!');
        break;
      case 'about':
        console.log('About feature - to be implemented');
        insertTextIntoInput('Slack Robot AI Assistant v1.0 - Powered by multiple LLM providers');
        break;
    }
  }

  async function handleRobotClick(event) {
    console.log('Robot button clicked!', event);
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    
    const button = event.currentTarget;
    
    // Get enhanced message data
    const messageData = getEnhancedMessageData();
    console.log('Enhanced message data:', messageData);
    
    // Check if we have content
    if (!messageData.content || messageData.content.trim() === '') {
      console.warn('No message content found, looking for messages in view...');
      // If no content, try to get the last visible message
      const lastMessage = getLastMessageContent();
      if (lastMessage && lastMessage.content) {
        messageData.content = lastMessage.content;
        messageData.user = lastMessage.user;
        console.log('Found last message:', lastMessage);
      } else {
        console.error('Could not find any messages in the current view');
        insertTextIntoInput('Could not find any messages to respond to. Please make sure there are messages visible in the channel.');
        button.style.opacity = '1';
        button.disabled = false;
        return;
      }
    }
    
    // Add loading state
    button.style.opacity = '0.5';
    button.disabled = true;
    
    // Fetch suggestion from API
    console.log('Fetching enhanced suggestion from API...');
    insertTextIntoInput('Fetching suggestion...');
    
    const message = await fetchSuggestion(messageData);
    
    console.log('Inserting message:', message);
    insertTextIntoInput(message);
    
    // Remove loading state
    button.style.opacity = '1';
    button.disabled = false;
  }

  async function handleLLMSettingsClick(event) {
    console.log('LLM Settings button clicked!');
    event.preventDefault();
    event.stopPropagation();
    
    createLLMModal();
  }

  function injectRobotButton() {
    console.log('=== injectRobotButton called ===');
    
    // Look for the main message composer buttons area
    // Primary selector based on the provided HTML structure
    let textyButtonsContainers = document.querySelectorAll('.c-texty_buttons');
    
    if (textyButtonsContainers.length === 0) {
      console.log('No .c-texty_buttons found, trying data-qa selector...');
      textyButtonsContainers = document.querySelectorAll('[data-qa="texty_composer_buttons"]');
    }
    
    if (textyButtonsContainers.length === 0) {
      console.log('Still no buttons container found, trying to find via message input...');
      
      // Fallback: Look for the main composer button area via message input
      const messageInput = document.querySelector('[data-qa="message_input"]');
      if (messageInput) {
        // Look for button container near the message input
        const parent = messageInput.closest('.c-texty_input_unstyled') || messageInput.parentElement;
        if (parent) {
          const buttonContainer = parent.querySelector('.c-texty_buttons') ||
                                parent.querySelector('.c-wysiwyg_container__formatting_buttons') ||
                                parent.querySelector('[role="toolbar"]');
          if (buttonContainer) {
            console.log('Found button container via message input parent');
            textyButtonsContainers = [buttonContainer];
          }
        }
      }
    }
    
    textyButtonsContainers.forEach((container, index) => {
      if (!container.querySelector('[data-qa="robot-button"]')) {
        console.log(`Processing texty buttons container ${index}`);
        
        // Use the container itself as the button row
        const buttonRow = container;
        
        console.log('Using button container:', buttonRow);
        
        // Look for a good reference button to insert after
        // Based on the provided HTML, we can look for specific buttons
        const referenceButton = buttonRow.querySelector('button[data-qa="slash_commands_composer_button"]') ||
                               buttonRow.querySelector('button[data-qa="audio_composer_button"]') ||
                               buttonRow.querySelector('button[data-qa="video_composer_button"]') ||
                               buttonRow.querySelector('.c-wysiwyg_container__footer_divider:last-of-type') ||
                               Array.from(buttonRow.querySelectorAll('button')).pop();
        
        // Add robot button
        const robotButton = createRobotButton();
        robotButton.addEventListener('click', handleRobotClick);
        
        // Add right-click context menu
        robotButton.addEventListener('contextmenu', (e) => {
          e.preventDefault();
          e.stopPropagation();
          createRobotContextMenu(e.clientX, e.clientY);
        });
        
        if (referenceButton && referenceButton.parentNode === buttonRow) {
          buttonRow.insertBefore(robotButton, referenceButton.nextSibling);
        } else if (referenceButton) {
          referenceButton.parentNode.appendChild(robotButton);
        } else {
          buttonRow.appendChild(robotButton);
        }
        console.log('Robot button added to texty buttons');
        
        // Add LLM settings button
        const settingsButton = createLLMSettingsButton();
        settingsButton.addEventListener('click', handleLLMSettingsClick);
        
        // Insert settings button after robot button
        robotButton.parentNode.insertBefore(settingsButton, robotButton.nextSibling);
        console.log('LLM Settings button added to texty buttons');
      }
    });
    
    if (textyButtonsContainers.length === 0) {
      console.log('Warning: Could not find texty_composer_buttons area');
    }
    
    console.log('=== injectRobotButton completed ===');
  }

  function setupObserver() {
    console.log('Setting up mutation observer...');
    observer = new MutationObserver((mutations) => {
      let shouldInject = false;
      
      for (const mutation of mutations) {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
          for (const node of mutation.addedNodes) {
            if (node.nodeType === Node.ELEMENT_NODE) {
              // Only watch for texty buttons containers, not sticky bar
              if (node.matches && (
                node.matches('.c-texty_buttons') ||
                node.matches('[data-qa="texty_composer_buttons"]') ||
                node.matches('[data-qa="message_input"]') ||
                node.matches('.c-texty_input_unstyled')
              ) || node.querySelector && (
                node.querySelector('.c-texty_buttons') ||
                node.querySelector('[data-qa="texty_composer_buttons"]') ||
                node.querySelector('[data-qa="message_input"]')
              )) {
                shouldInject = true;
                break;
              }
            }
          }
        }
      }
      
      if (shouldInject) {
        console.log('New message composer detected, injecting buttons...');
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
