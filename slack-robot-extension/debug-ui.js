// Debug script to help identify Slack UI elements
// Run this in the browser console while on Slack

console.log('=== Slack UI Debug ===');

// Check for message composer areas
const composerSelectors = [
  '.p-texty_sticky_formatting_bar',
  '[data-qa="texty_composer_buttons"]',
  '.c-wysiwyg_container__formatting_buttons',
  '.c-texty_buttons__container',
  '[data-qa="message_input"]',
  '.ql-editor',
  '.p-composer__body',
  '.c-composer__button_menu',
  '[role="toolbar"]',
  '.p-composer_page__input',
  '.p-wysiwyg_input__buttons'
];

console.log('\n1. Looking for composer elements:');
composerSelectors.forEach(selector => {
  const elements = document.querySelectorAll(selector);
  if (elements.length > 0) {
    console.log(`✓ Found ${elements.length} element(s) with selector: ${selector}`, elements);
  }
});

// Check for existing buttons
const buttonSelectors = [
  'button[data-qa="texty_code_button"]',
  'button[data-qa="texty_code_block_button"]',
  'button[data-qa="code-block-composer-button"]',
  'button[data-qa="code-composer-button"]',
  'button[aria-label*="format"]',
  'button[aria-label*="emoji"]',
  'button[aria-label*="mention"]',
  '.c-icon_button',
  '.c-wysiwyg_container__button'
];

console.log('\n2. Looking for existing buttons:');
buttonSelectors.forEach(selector => {
  const elements = document.querySelectorAll(selector);
  if (elements.length > 0) {
    console.log(`✓ Found ${elements.length} button(s) with selector: ${selector}`, elements);
  }
});

// Check for our injected buttons
console.log('\n3. Looking for our buttons:');
const robotButton = document.querySelector('[data-qa="robot-button"]');
const llmButton = document.querySelector('[data-qa="llm-settings-button"]');
console.log('Robot button:', robotButton || 'NOT FOUND');
console.log('LLM Settings button:', llmButton || 'NOT FOUND');

// Check the actual button area structure
console.log('\n4. Analyzing button area structure:');
const messageInput = document.querySelector('[data-qa="message_input"]');
if (messageInput) {
  console.log('Message input found:', messageInput);
  console.log('Parent structure:', messageInput.parentElement);
  console.log('Siblings:', Array.from(messageInput.parentElement.children));
}

// Look for any toolbar or button container
console.log('\n5. Looking for any toolbar or button container:');
const toolbars = document.querySelectorAll('[role="toolbar"], .c-texty_input__button_container, .p-composer__buttons');
toolbars.forEach((toolbar, i) => {
  console.log(`Toolbar ${i}:`, toolbar);
  console.log('  Buttons inside:', toolbar.querySelectorAll('button'));
});

console.log('\n=== End Debug ===');