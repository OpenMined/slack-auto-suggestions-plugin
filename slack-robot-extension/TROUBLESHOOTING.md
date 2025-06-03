# Troubleshooting: Buttons Not Appearing

## Quick Check Steps

1. **Verify Extension is Loaded**
   - Open Chrome Developer Tools (F12) in Slack
   - Go to Console tab
   - You should see logs like:
     ```
     Slack Robot Assistant initializing...
     === injectRobotButton called ===
     ```

2. **Check for Errors**
   - Look for any red error messages in the console
   - Common errors:
     - "Cannot read property..." - UI structure changed
     - "Refused to connect..." - API server not running

3. **Manual Debug**
   - Copy and paste this into the console:
   ```javascript
   // Check if content script is loaded
   console.log('Extension loaded:', typeof injectRobotButton !== 'undefined');
   
   // Try to manually inject buttons
   if (typeof injectRobotButton !== 'undefined') {
     injectRobotButton();
   }
   
   // Look for button areas
   const areas = [
     '[data-qa="texty_composer_buttons"]',
     '.c-wysiwyg_container__formatting_buttons',
     '[role="toolbar"]'
   ];
   areas.forEach(s => {
     const els = document.querySelectorAll(s);
     if (els.length) console.log(`Found ${els.length} ${s}`, els);
   });
   ```

4. **Check Extension Permissions**
   - Go to chrome://extensions
   - Click on "Slack Robot Assistant" details
   - Ensure "Site access" includes slack.com

## Common Issues

### Issue 1: Buttons not appearing at all
**Solution**: The Slack UI might have changed. Run the debug script:
1. Open `/slack-robot-extension/debug-ui.js` 
2. Copy its contents
3. Paste in Slack's console
4. Share the output - it will show where buttons should go

### Issue 2: "Unknown action" errors
**Solution**: Already fixed - old message capture code was removed

### Issue 3: API connection errors
**Solution**: 
- Ensure API server is running: `cd api-server && python main.py`
- Check it's accessible: `curl http://localhost:8000/health`

### Issue 4: Buttons appear but don't work
**Solution**: Check console for errors when clicking. May need to:
- Configure an LLM provider first
- Check API server logs

## Next Steps if Still Not Working

1. **Reload Everything**
   ```
   1. Stop the API server (Ctrl+C)
   2. Reload extension in chrome://extensions
   3. Start API server: python main.py
   4. Hard refresh Slack (Ctrl+Shift+R)
   5. Open a DM or channel and look for buttons
   ```

2. **Check Specific Message Composer**
   - Click in the message input field
   - Look for formatting buttons (Bold, Italic, etc.)
   - Our buttons should appear after those

3. **Try Different Slack Views**
   - Direct Messages
   - Channels
   - Threads
   - The button injection might work in some but not others

## Developer Console Commands

```javascript
// Force re-initialization
if (typeof initialize !== 'undefined') initialize();

// Check current provider status
chrome.runtime.sendMessage({action: 'getCurrentProvider'}, r => console.log('Provider:', r));

// List all buttons in composer area
document.querySelectorAll('.c-texty_input button').forEach(b => 
  console.log(b.getAttribute('aria-label') || b.textContent)
);
```