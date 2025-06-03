(function() {
  'use strict';

  let observer = null;
  let processedMessages = new Set();
  let currentConversation = null;
  
  // Configuration
  const CONFIG = {
    captureHistory: true, // Enable/disable history capture
    minMessagesThreshold: 10, // Minimum messages to consider conversation "saved"
    maxHistoryScrollAttempts: 3, // Max times to scroll up for more history
    historyCaptureThreshold: 50, // Capture history if less than this many messages
    debug: true // Enable debug logging
  };

  // Extract conversation info from URL and page
  function getConversationInfo() {
    const url = window.location.href;
    
    // Try different URL patterns
    // Pattern 1: /archives/CHANNEL_ID
    // Pattern 2: /client/TEAM_ID/CHANNEL_ID
    // Pattern 3: /messages/CHANNEL_ID
    // Pattern 4: Slack app URLs with hash routing
    let conversationId = null;
    
    // Try archives pattern first
    let match = url.match(/\/archives\/([A-Z0-9]+)/);
    if (match) {
      conversationId = match[1];
    } else {
      // Try client pattern
      match = url.match(/\/client\/[A-Z0-9]+\/([A-Z0-9]+)/);
      if (match) {
        conversationId = match[1];
      } else {
        // Try messages pattern
        match = url.match(/\/messages\/([A-Z0-9]+)/);
        if (match) {
          conversationId = match[1];
        } else {
          // Try hash-based routing (Slack app)
          match = url.match(/#\/([A-Z0-9]+)/);
          if (match) {
            conversationId = match[1];
          }
        }
      }
    }
    
    // Also check for conversation ID in data attributes
    if (!conversationId) {
      // Try to find from message links in the page
      const messageLink = document.querySelector('a[href*="/archives/"]');
      if (messageLink) {
        const linkMatch = messageLink.href.match(/\/archives\/([A-Z0-9]+)/);
        if (linkMatch) {
          conversationId = linkMatch[1];
          console.log('Found conversation ID from message link:', conversationId);
        }
      }
    }
    
    // Try to get from data attributes
    if (!conversationId) {
      const channelElement = document.querySelector('[data-channel-id]');
      if (channelElement) {
        conversationId = channelElement.getAttribute('data-channel-id');
        console.log('Found conversation ID from data attribute:', conversationId);
      }
    }
    
    if (!conversationId) {
      console.log('Could not extract conversation ID from URL:', url);
      
      // Debug: Log what we can find in the page
      if (CONFIG.debug) {
        console.log('Debug: Looking for conversation indicators...');
        console.log('- URL:', url);
        console.log('- Title:', document.title);
        console.log('- Hash:', window.location.hash);
        console.log('- Pathname:', window.location.pathname);
        
        // Log any elements with channel/conversation data
        const debugSelectors = [
          '[data-channel]',
          '[data-conversation-id]',
          '[data-qa*="channel"]',
          '[data-qa*="conversation"]',
          '[aria-label*="channel"]',
          '.p-channel_sidebar__channel--selected'
        ];
        
        debugSelectors.forEach(selector => {
          const elements = document.querySelectorAll(selector);
          if (elements.length > 0) {
            console.log(`- Found ${elements.length} elements with selector "${selector}":`, elements);
          }
        });
      }
      
      return null;
    }
    
    // Try multiple selectors for channel name
    const titleSelectors = [
      '[data-qa="channel_name"]',
      '[data-qa="channel_title"]', 
      '.p-view_header__title',
      '.p-classic_channel_header__title',
      '.p-channel_sidebar__name',
      '[data-qa="channel_sidebar_name_button"]',
      '.c-channel_entity__name',
      '.p-channel_header__title'
    ];
    
    let titleElement = null;
    for (const selector of titleSelectors) {
      titleElement = document.querySelector(selector);
      if (titleElement) break;
    }
    
    const name = titleElement ? titleElement.textContent.trim() : conversationId;
    
    // Determine conversation type
    let type = 'channel';
    if (conversationId.startsWith('D')) {
      type = 'dm';
    } else if (conversationId.startsWith('G')) {
      type = 'group_dm';
    }
    
    // Get workspace ID
    const workspaceMatch = url.match(/https:\/\/([^.]+)\.slack\.com/);
    const workspaceId = workspaceMatch ? workspaceMatch[1] : 'unknown';
    
    console.log('Detected conversation:', { id: conversationId, name, type, workspace_id: workspaceId });
    
    return {
      id: conversationId,
      name: name,
      type: type,
      workspace_id: workspaceId
    };
  }

  // Extract message data from message element with retry for user info
  async function extractMessageData(messageElement, retryCount = 0) {
    if (!messageElement || !messageElement.id) return null;
    
    const tsMatch = messageElement.id.match(/message-list_(\d+\.\d+)/);
    if (!tsMatch) return null;
    
    const ts = tsMatch[1];
    
    // Skip if already processed
    if (processedMessages.has(ts)) return null;
    
    // Skip if no current conversation
    if (!currentConversation || !currentConversation.id) return null;
    
    const gutter = messageElement.querySelector('.c-message_kit__gutter__right');
    if (!gutter) return null;
    
    // Extract user info with multiple selector attempts
    let senderButton = gutter.querySelector('[data-message-sender]');
    let userId = 'unknown';
    let userName = 'Unknown User';
    
    // Try multiple selectors for user info
    const userSelectors = [
      '[data-message-sender]',
      '.c-message__sender_button',
      'button[data-qa="message_sender_name"]',
      '.c-message_kit__sender',
      'a[data-qa="message_sender_link"]'
    ];
    
    for (const selector of userSelectors) {
      senderButton = gutter.querySelector(selector);
      if (senderButton) {
        userId = senderButton.getAttribute('data-message-sender') || 
                senderButton.getAttribute('data-user-id') || 
                senderButton.getAttribute('data-userid') || 
                'unknown';
        userName = senderButton.textContent.trim() || 'Unknown User';
        if (userId !== 'unknown' && userName !== 'Unknown User') break;
      }
    }
    
    // If still no user info and we haven't retried too many times, wait and retry
    if ((userId === 'unknown' || userName === 'Unknown User') && retryCount < 3) {
      await new Promise(resolve => setTimeout(resolve, 200 * (retryCount + 1))); // Progressive delay
      return extractMessageData(messageElement, retryCount + 1);
    }
    
    // Log if we still couldn't find user info after retries
    if (userId === 'unknown' || userName === 'Unknown User') {
      console.warn(`Could not extract user info for message ${ts} after ${retryCount + 1} attempts`);
      if (CONFIG.debug) {
        console.log('Message element HTML:', messageElement.outerHTML);
      }
    }
    
    // Extract message text
    const messageBlocks = gutter.querySelector('.c-message__message_blocks');
    const text = messageBlocks ? messageBlocks.textContent.trim() : '';
    
    // Skip empty messages
    if (!text && !gutter.querySelector('.c-message_kit__attachments')) return null;
    
    // Check for attachments
    const hasAttachments = !!gutter.querySelector('.c-message_kit__attachments');
    
    // Extract reactions
    const reactionElements = gutter.querySelectorAll('.c-reaction');
    const reactions = Array.from(reactionElements).map(reaction => {
      const emoji = reaction.querySelector('[data-stringify-emoji]');
      const count = reaction.querySelector('.c-reaction__count');
      return {
        emoji: emoji ? emoji.getAttribute('data-stringify-emoji') : '',
        count: count ? parseInt(count.textContent) : 0
      };
    });
    
    // Check if in thread
    const threadTs = messageElement.querySelector('[data-qa="thread_message_reply_count"]') ? ts : null;
    
    // Store raw HTML for attachments/rich content
    const rawHtml = hasAttachments ? gutter.innerHTML : null;
    
    return {
      ts: ts,
      conversation_id: currentConversation.id,
      user_id: userId,
      user_name: userName,
      text: text,
      raw_html: rawHtml,
      has_attachments: hasAttachments,
      reactions: reactions.length > 0 ? JSON.stringify(reactions) : null,
      thread_ts: threadTs
    };
  }

  // Send data to API
  async function sendToAPI(endpoint, data) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { 
          action: 'apiRequest',
          endpoint: endpoint,
          method: 'POST',
          data: data
        },
        (response) => {
          if (chrome.runtime.lastError) {
            console.error('Chrome runtime error:', chrome.runtime.lastError);
            resolve({ success: false, error: chrome.runtime.lastError.message });
            return;
          }
          resolve(response || { success: false });
        }
      );
    });
  }

  // Process new messages
  async function processNewMessages() {
    // Skip if no current conversation
    if (!currentConversation) {
      console.log('No current conversation, skipping message processing');
      return;
    }
    
    const messageElements = document.querySelectorAll('[id^="message-list_"][role="listitem"]');
    const messages = [];
    
    for (const element of messageElements) {
      try {
        const messageData = await extractMessageData(element); // Now async
        if (messageData) {
          messages.push(messageData);
          processedMessages.add(messageData.ts);
        }
      } catch (error) {
        console.error('Error extracting message data:', error, element);
      }
    }
    
    if (messages.length > 0) {
      console.log(`Capturing ${messages.length} new messages`);
      
      // Send in batches of 50
      for (let i = 0; i < messages.length; i += 50) {
        const batch = messages.slice(i, i + 50);
        const response = await sendToAPI('/api/messages/bulk', { messages: batch });
        
        if (response.success) {
          console.log(`Saved batch of ${batch.length} messages`);
        } else {
          console.error('Failed to save messages:', response.error);
        }
      }
    }
  }

  // Check if conversation has existing messages
  async function checkConversationHistory(conversationId) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { 
          action: 'apiRequest',
          endpoint: `/api/conversations/${conversationId}/has-messages`,
          method: 'GET'
        },
        (response) => {
          if (chrome.runtime.lastError || !response?.success) {
            console.error('Failed to check conversation history:', chrome.runtime.lastError);
            resolve({ exists: false, message_count: 0 });
            return;
          }
          resolve(response.data || { exists: false, message_count: 0 });
        }
      );
    });
  }

  // Capture all visible messages (for history)
  async function captureAllVisibleMessages() {
    // Skip if no current conversation
    if (!currentConversation) {
      console.log('No current conversation, skipping history capture');
      return;
    }
    
    console.log('Capturing all visible messages for history...');
    
    // Wait a bit for messages to fully load
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const messageElements = document.querySelectorAll('[id^="message-list_"][role="listitem"]');
    const messages = [];
    
    for (const element of messageElements) {
      try {
        const messageData = await extractMessageData(element); // Now async
        if (messageData) {
          messages.push(messageData);
          // Mark as processed to avoid re-processing in normal flow
          processedMessages.add(messageData.ts);
        }
      } catch (error) {
        console.error('Error extracting message data:', error, element);
      }
    }
    
    if (messages.length > 0) {
      console.log(`Found ${messages.length} messages to capture for history`);
      
      // Send in batches of 50
      for (let i = 0; i < messages.length; i += 50) {
        const batch = messages.slice(i, i + 50);
        const response = await sendToAPI('/api/messages/bulk', { messages: batch });
        
        if (response.success) {
          console.log(`Saved history batch of ${batch.length} messages`);
        } else {
          console.error('Failed to save history messages:', response.error);
        }
      }
    } else {
      console.log('No messages found to capture');
    }
  }

  // Initialize conversation tracking
  async function initializeConversation() {
    const conversationInfo = getConversationInfo();
    if (!conversationInfo) {
      console.log('Could not determine conversation info');
      return;
    }
    
    if (currentConversation?.id !== conversationInfo.id) {
      currentConversation = conversationInfo;
      processedMessages.clear();
      
      console.log('Tracking conversation:', conversationInfo);
      
      // Create conversation in database
      const response = await sendToAPI('/api/conversations', conversationInfo);
      if (response.success) {
        console.log('Conversation registered:', conversationInfo.id);
        
        // Check if this conversation has saved messages
        try {
          const historyCheck = await checkConversationHistory(conversationInfo.id);
          console.log('History check result:', historyCheck);
          
          // If no messages or very few messages, capture all visible ones
          if (CONFIG.captureHistory && (!historyCheck.exists || historyCheck.message_count < CONFIG.minMessagesThreshold)) {
            console.log('Conversation has little/no history, capturing all visible messages...');
            await captureAllVisibleMessages();
          } else {
            console.log(`Conversation already has ${historyCheck.message_count} messages saved`);
          }
        } catch (error) {
          console.error('Error checking conversation history:', error);
          // If we can't check history, capture messages anyway
          console.log('Could not check history, capturing visible messages...');
          await captureAllVisibleMessages();
        }
      }
    }
  }

  // Setup observer for new messages
  function setupObserver() {
    if (observer) {
      observer.disconnect();
    }

    // Try multiple selectors for message list
    const messageListSelectors = [
      '#message-list',
      '[data-qa="message_list"]',
      '.c-message_list',
      '[role="list"][aria-label*="Messages"]',
      '.c-virtual_list__scroll_container'
    ];
    
    let messageList = null;
    for (const selector of messageListSelectors) {
      messageList = document.querySelector(selector);
      if (messageList) {
        console.log('Found message list with selector:', selector);
        break;
      }
    }
    
    if (!messageList) {
      console.log('Message list not found, retrying...');
      setTimeout(setupObserver, 2000);
      return;
    }

    observer = new MutationObserver((mutations) => {
      let hasNewMessages = false;
      
      for (const mutation of mutations) {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
          for (const node of mutation.addedNodes) {
            if (node.nodeType === 1 && 
                (node.id?.startsWith('message-list_') || 
                 node.classList?.contains('c-message_kit__background') ||
                 node.querySelector?.('[id^="message-list_"]'))) {
              hasNewMessages = true;
              break;
            }
          }
        }
      }
      
      if (hasNewMessages) {
        // Add a small delay to allow Slack to fully populate the DOM
        setTimeout(() => {
          processNewMessages();
        }, 500);
      }
    });

    observer.observe(messageList, {
      childList: true,
      subtree: true
    });

    console.log('Message observer initialized');
  }

  // Try to load more history by scrolling up
  async function loadMoreHistory() {
    const messageList = document.querySelector('#message-list');
    if (!messageList) return;
    
    const scrollContainer = messageList.closest('.c-virtual_list');
    if (!scrollContainer) return;
    
    // Check if we're already at the top
    const initialScrollTop = scrollContainer.scrollTop;
    if (initialScrollTop < 100) {
      console.log('Already at top of conversation, no more history to load');
      return false;
    }
    
    console.log('Attempting to load more history by scrolling up...');
    
    // Scroll to top to trigger loading more messages
    scrollContainer.scrollTo(0, 0);
    
    // Wait for new messages to load
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Check if new messages were loaded
    const newScrollHeight = scrollContainer.scrollHeight;
    return newScrollHeight > initialScrollTop;
  }

  // Wait for Slack to load
  async function waitForSlackToLoad() {
    console.log('Waiting for Slack to load...');
    let attempts = 0;
    const maxAttempts = 20;
    
    while (attempts < maxAttempts) {
      // Check if we can find any message elements or Slack UI elements
      const slackLoaded = document.querySelector('[id^="message-list_"]') || 
                         document.querySelector('.c-message_kit__background') ||
                         document.querySelector('[data-qa="message_content"]') ||
                         document.querySelector('.p-workspace__primary_view');
      
      if (slackLoaded) {
        console.log('Slack appears to be loaded');
        return true;
      }
      
      attempts++;
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    console.log('Timeout waiting for Slack to load');
    return false;
  }

  // Handle page changes
  async function handlePageChange() {
    console.log('Page changed, reinitializing...');
    
    // Wait for Slack to actually load
    const loaded = await waitForSlackToLoad();
    if (!loaded) {
      console.log('Slack did not load properly, will retry on next change');
      return;
    }
    
    await initializeConversation();
    setupObserver();
    
    // Check if we should try to load full history
    if (currentConversation) {
      const historyCheck = await checkConversationHistory(currentConversation.id);
      
      // If conversation has very few messages, try to load more history
      if (CONFIG.captureHistory && (!historyCheck.exists || historyCheck.message_count < CONFIG.historyCaptureThreshold)) {
        console.log('Attempting to load full conversation history...');
        
        // Try to load more history up to configured attempts
        for (let i = 0; i < CONFIG.maxHistoryScrollAttempts; i++) {
          const loadedMore = await loadMoreHistory();
          if (!loadedMore) break;
          
          // Capture newly loaded messages
          await captureAllVisibleMessages();
        }
      }
    }
  }

  // Initialize
  function initialize() {
    console.log('Slack message capture initializing...');
    
    // Watch for URL changes
    let lastUrl = window.location.href;
    setInterval(() => {
      const currentUrl = window.location.href;
      if (currentUrl !== lastUrl) {
        lastUrl = currentUrl;
        handlePageChange();
      }
    }, 1000);
    
    // Initial setup with delay for Slack to load
    setTimeout(() => {
      handlePageChange();
    }, 2000);
  }

  // Start when page is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
  } else {
    initialize();
  }

  // Cleanup on page unload
  window.addEventListener('beforeunload', () => {
    if (observer) {
      observer.disconnect();
    }
  });

  console.log('Slack message capture script loaded');
})();