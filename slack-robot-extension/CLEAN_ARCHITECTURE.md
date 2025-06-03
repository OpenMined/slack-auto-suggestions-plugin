# Slack Robot Assistant - Clean Architecture

## Overview
The Slack Robot Assistant is an AI-powered Chrome extension that adds intelligent message suggestions to Slack using configurable LLM providers.

## Core Features
1. **Robot Button**: Inserts AI-generated suggestions into message composer
2. **LLM Settings**: Configure and switch between different AI providers
3. **Context Awareness**: Sends conversation context for better suggestions

## File Structure
```
slack-robot-extension/
├── manifest.json          # Extension configuration
├── background.js          # Handles API communication
├── content/
│   └── slack-robot.js     # Main UI and functionality
│   └── slack-robot.css    # Styling
└── icons/                 # Extension icons
```

## Removed Features
- Message capture functionality (slack-message-capture.js) - Not needed for core assistant
- Unlimited storage permission - Not required
- Multiple content scripts - Simplified to one

## Key Components

### 1. Content Script (slack-robot.js)
- Injects robot button into Slack message composer
- Adds LLM settings button with status indicator
- Extracts conversation context (ID, thread, user)
- Handles button clicks and modal UI

### 2. Background Script (background.js)
- Communicates with API server
- Handles all LLM provider operations:
  - fetchSuggestion
  - getCurrentProvider
  - getProviderInfo
  - configureProvider
  - activateProvider
  - testProvider

### 3. API Integration
- Endpoint: http://localhost:8000
- Enhanced suggestion endpoint with context
- Provider management endpoints

## Data Flow
1. User clicks robot button
2. Content script extracts context
3. Background script calls API
4. API returns AI suggestion
5. Content script inserts text

## Next Steps
1. Reload extension in Chrome
2. Refresh Slack
3. Configure an LLM provider
4. Start using AI suggestions!