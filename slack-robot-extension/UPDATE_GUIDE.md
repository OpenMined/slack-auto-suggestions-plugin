# Slack Robot Extension Update Guide

## How to Update the Extension

The extension has been enhanced with LLM provider management features. To use the new version:

### 1. Replace Files

Replace these files with their enhanced versions:
- `content/slack-robot.js` → `content/slack-robot-enhanced.js`
- `background.js` → `background-enhanced.js`

### 2. Update manifest.json

Update the manifest to use the new files:

```json
{
  "manifest_version": 3,
  "name": "Slack Robot Button",
  "version": "2.0",  // Updated version
  "description": "Slack assistant with LLM provider management",
  "permissions": [
    "activeTab",
    "storage",
    "unlimitedStorage"
  ],
  "host_permissions": [
    "https://*.slack.com/*",
    "http://localhost:8000/*",
    "http://127.0.0.1:8000/*"
  ],
  "background": {
    "service_worker": "background-enhanced.js"  // Updated
  },
  "content_scripts": [
    {
      "matches": ["https://*.slack.com/*"],
      "js": ["content/slack-robot-enhanced.js", "content/slack-message-capture.js"],  // Updated
      "css": ["content/slack-robot.css"]
    }
  ],
  "icons": {
    "16": "icons/robot16.png",
    "48": "icons/robot48.png",
    "128": "icons/robot128.png"
  }
}
```

### 3. Reload Extension

1. Go to `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Reload" on the Slack Robot extension

## New Features

### 1. LLM Settings Button
- Look for the settings icon (⚙️) with a status indicator next to the robot button
- Green dot = LLM active
- Red dot = LLM error
- Yellow dot = LLM loading
- Gray dot = LLM inactive

### 2. LLM Provider Modal
Click the settings button to:
- View current provider status
- See all configured providers
- Configure new providers
- Switch between providers
- Test provider connections

### 3. Enhanced Context
The extension now sends:
- `conversation_id` - Current channel/DM ID
- `thread_ts` - Thread timestamp (if in a thread)
- `user_id` - User identifier
- Complete message context

## Usage

### Configure a Provider

1. Click the LLM Settings button (⚙️)
2. In the modal, select a provider:
   - **Anthropic**: Requires API key
   - **OpenAI**: Requires API key
   - **Ollama**: Requires base URL (default: http://localhost:11434)
   - **OpenRouter**: Requires API key and optionally base URL
3. Enter the model name (e.g., `gpt-3.5-turbo`, `claude-3-haiku-20240307`)
4. Enter API key if required
5. Click "Configure"
6. Click "Test" to verify connection

### Activate a Provider

1. In the providers list, click "Activate" next to any configured provider
2. The status indicator will turn green when active

### Get Suggestions

1. Click the robot button as usual
2. The suggestion will now use:
   - Vector search for similar messages
   - Conversation history (if available)
   - Thread context (if in a thread)
   - LLM enhancement (if configured)

## Troubleshooting

### No LLM Response
- Check if a provider is configured and active
- Verify API keys are correct
- Test the provider connection
- Check if the API server is running

### Slow Responses
- Normal LLM response time: 1-3 seconds
- Check your internet connection
- Try a different provider

### Extension Not Working
1. Make sure the API server is running: `python main.py`
2. Check browser console for errors (F12)
3. Reload the extension
4. Clear browser cache and reload Slack

## API Requirements

The API server must have these endpoints:
- `POST /suggestion` - Enhanced with context parameters
- `GET /api/llm/providers` - List providers
- `POST /api/llm/providers/configure` - Configure provider
- `POST /api/llm/providers/activate` - Activate provider
- `POST /api/llm/providers/test` - Test provider
- `GET /api/llm/current` - Get current provider