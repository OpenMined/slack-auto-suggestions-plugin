# Message Capture Functionality Restored

## What was fixed
The message capture functionality was accidentally removed during the cleanup. It has now been restored while keeping the robot assistant features.

## Changes made

1. **manifest.json**
   - Added `slack-message-capture.js` back as a second content script
   - Set to run at `document_idle` to avoid conflicts

2. **background.js**
   - Added `apiRequest` action handler back
   - Handles all message capture API calls to the backend

## How it works now

The extension now has two independent features:

1. **Message Capture** (slack-message-capture.js)
   - Automatically captures Slack messages as you browse
   - Stores them in the vector database
   - Runs in the background without user interaction
   - Captures conversation history when you enter a channel

2. **Robot Assistant** (slack-robot.js)
   - Provides the robot button for AI suggestions
   - Manages LLM provider settings
   - Uses the captured messages for context-aware responses

## To apply changes

1. Reload the extension in Chrome (chrome://extensions)
2. Refresh Slack
3. Messages should start being captured automatically
4. The robot button will use these captured messages for better suggestions

## Architecture

```
Slack Page
    ├── slack-message-capture.js (captures messages)
    │   └── Sends to API → Vector Database
    │
    └── slack-robot.js (AI suggestions)
        └── Queries API → Vector Database + LLM
```

Both features work independently but complement each other.