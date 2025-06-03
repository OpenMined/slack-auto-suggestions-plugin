# Slack Robot Button Extension

This Chrome extension adds a robot icon to Slack's message composer that inserts custom text when clicked.

## Features

- Adds a robot icon to the sticky formatting bar (appears when text is selected)
- Adds a robot icon to the bottom texty buttons bar
- Clicking the robot icon inserts "Hello World {context}" where context is either "sticky_bar" or "texty_buttons"

## Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" in the top right corner
3. Click "Load unpacked"
4. Select the `slack-robot-extension` folder
5. The extension should now be active

## Usage

1. Navigate to any Slack workspace (https://*.slack.com)
2. Open a channel or direct message
3. You'll see a robot icon in two places:
   - In the formatting toolbar that appears when you select text
   - In the bottom toolbar next to the slash command button
4. Click the robot icon to insert the text into the message field

## File Structure

- `manifest.json` - Extension configuration
- `content/slack-robot.js` - Main content script that injects the robot buttons
- `content/slack-robot.css` - Styles for the robot buttons
- `icons/` - Extension icons (placeholder icons included)

## Notes

- The extension automatically detects when Slack's UI updates and re-injects the buttons as needed
- The text insertion properly handles Slack's rich text editor
- Works with both light and dark themes