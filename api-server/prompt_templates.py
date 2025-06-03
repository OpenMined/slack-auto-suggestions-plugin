"""
Advanced prompt templates for enhanced suggestion generation
"""
from typing import Dict, List, Optional
from datetime import datetime
import re


class PromptTemplates:
    """Collection of prompt templates for different conversation contexts"""
    
    # Base template for all suggestions
    BASE_TEMPLATE = """You are an AI assistant helping to generate message suggestions for Slack conversations.

{context_section}

Current message from {user_name}:
"{current_message}"

{additional_instructions}

Generate a helpful response that:
- Addresses the topic directly and naturally
- Maintains the appropriate tone based on the context
- Is concise and suitable for Slack (1-3 sentences max)
- Fits the conversation style shown in the context

Response:"""

    # Context-specific templates
    TEMPLATES = {
        "question": {
            "instructions": "The user is asking a question. Provide a helpful, informative response.",
            "tone": "helpful and informative"
        },
        "technical_discussion": {
            "instructions": "This appears to be a technical discussion. Provide a technically accurate and professional response.",
            "tone": "professional and precise"
        },
        "project_update": {
            "instructions": "The user is discussing project status or updates. Respond appropriately to acknowledge or follow up.",
            "tone": "professional and engaged"
        },
        "casual_chat": {
            "instructions": "This is a casual conversation. Keep the response friendly and conversational.",
            "tone": "friendly and relaxed"
        },
        "request": {
            "instructions": "The user is making a request. Acknowledge the request and respond appropriately.",
            "tone": "responsive and helpful"
        },
        "feedback": {
            "instructions": "The user is providing or seeking feedback. Respond constructively.",
            "tone": "constructive and appreciative"
        },
        "scheduling": {
            "instructions": "This involves scheduling or time coordination. Be clear about availability and timing.",
            "tone": "clear and accommodating"
        },
        "default": {
            "instructions": "Generate an appropriate response based on the context.",
            "tone": "professional yet friendly"
        }
    }

    # Keywords for detecting conversation type
    CONVERSATION_PATTERNS = {
        "question": [
            r"\?$", r"^(what|where|when|why|how|who|which|can|could|would|should|is|are|do|does)",
            r"(wondering|curious|question|ask|know if)", r"(help|assist|explain|clarify)"
        ],
        "technical_discussion": [
            r"(bug|error|issue|problem|fix|debug|code|api|database|server|deploy|git|PR|pull request)",
            r"(function|method|class|variable|parameter|endpoint|query|performance)",
            r"(implement|refactor|optimize|test|review)"
        ],
        "project_update": [
            r"(update|status|progress|completed|finished|done|working on|started)",
            r"(milestone|sprint|task|ticket|story|epic)",
            r"(roadmap|timeline|deadline|launch|release)"
        ],
        "request": [
            r"(can you|could you|would you|please|need|want|require)",
            r"(request|asking|favor|help me|assist with)",
            r"(send|share|provide|give|show)"
        ],
        "feedback": [
            r"(feedback|review|thoughts|opinion|suggestion|improve)",
            r"(great|excellent|good job|well done|thanks|appreciate)",
            r"(concern|issue|problem with|not sure about)"
        ],
        "scheduling": [
            r"(meeting|call|sync|discuss|chat|connect)",
            r"(available|free|calendar|schedule|time|slot)",
            r"(tomorrow|today|next week|this week|morning|afternoon|evening)",
            r"(\d{1,2}:\d{2}|\d{1,2}(am|pm))"
        ]
    }

    @classmethod
    def detect_conversation_type(cls, message: str, context_messages: List[str] = None) -> str:
        """Detect the type of conversation based on message content and context"""
        message_lower = message.lower()
        
        # Limit context messages to prevent performance issues
        if context_messages:
            context_messages = context_messages[:3]  # Only use first 3 context messages
        
        # Check each pattern
        scores = {}
        for conv_type, patterns in cls.CONVERSATION_PATTERNS.items():
            score = 0
            for pattern in patterns:
                try:
                    if re.search(pattern, message_lower, re.IGNORECASE):
                        score += 1
                except re.error:
                    # Skip problematic regex patterns
                    continue
            
            # Also check context if provided (limited to prevent hanging)
            if context_messages:
                for ctx_msg in context_messages:
                    if not ctx_msg:  # Skip empty messages
                        continue
                    ctx_lower = ctx_msg.lower()[:200]  # Limit message length
                    for pattern in patterns:
                        try:
                            if re.search(pattern, ctx_lower, re.IGNORECASE):
                                score += 0.5
                                break  # Only count one match per message per type
                        except re.error:
                            continue
            
            scores[conv_type] = score
        
        # Return the type with highest score, or default
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]
        
        return "default"

    @classmethod
    def build_context_section(cls, similar_messages: List[Dict], 
                            conversation_history: List[Dict] = None,
                            thread_messages: List[Dict] = None,
                            knowledge_documents: List[Dict] = None) -> str:
        """Build the context section for the prompt"""
        sections = []
        
        # Add knowledge documents first (highest priority)
        if knowledge_documents:
            knowledge_context = "Relevant knowledge from uploaded documents:\n"
            for i, doc in enumerate(knowledge_documents[:3], 1):
                filename = doc.get('filename', 'Unknown')
                content = doc.get('content', '')
                chunk_index = doc.get('chunk_index', '0')
                similarity = doc.get('similarity_score', 0)
                
                # Include full content for LLM processing (don't truncate here)
                knowledge_context += f"{i}. From {filename} (chunk {chunk_index}, similarity: {similarity:.3f}):\n{content}\n\n"
            sections.append(knowledge_context.strip())
        
        # Add similar messages context
        if similar_messages:
            similar_context = "Similar past messages in this workspace:\n"
            for i, msg in enumerate(similar_messages[:3], 1):
                user = msg.get('user_name', 'Unknown')
                text = msg.get('text', '')
                similar_context += f"{i}. {user}: {text}\n"
            sections.append(similar_context.strip())
        
        # Add recent conversation history
        if conversation_history:
            history_context = "Recent conversation history:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages
                user = msg.get('user_name', 'Unknown')
                text = msg.get('text', '')
                history_context += f"{user}: {text}\n"
            sections.append(history_context.strip())
        
        # Add thread context
        if thread_messages:
            thread_context = "Thread context:\n"
            for msg in thread_messages[-3:]:  # Last 3 thread messages
                user = msg.get('user_name', 'Unknown')
                text = msg.get('text', '')
                thread_context += f"{user}: {text}\n"
            sections.append(thread_context.strip())
        
        return "\n\n".join(sections) if sections else "No previous context available."

    @classmethod
    def generate_prompt(cls, user_name: str, current_message: str,
                       similar_messages: List[Dict] = None,
                       conversation_history: List[Dict] = None,
                       thread_messages: List[Dict] = None,
                       conversation_type: str = None,
                       knowledge_documents: List[Dict] = None) -> str:
        """Generate a complete prompt based on the context and conversation type"""
        
        # Auto-detect conversation type if not provided
        if not conversation_type:
            context_texts = []
            if similar_messages:
                context_texts.extend([msg.get('text', '') for msg in similar_messages])
            if conversation_history:
                context_texts.extend([msg.get('text', '') for msg in conversation_history])
            
            conversation_type = cls.detect_conversation_type(current_message, context_texts)
        
        # Build context section including knowledge documents
        context_section = cls.build_context_section(
            similar_messages, conversation_history, thread_messages, knowledge_documents
        )
        
        # Get template instructions
        template_info = cls.TEMPLATES.get(conversation_type, cls.TEMPLATES["default"])
        additional_instructions = template_info["instructions"]
        
        # Add time-aware instructions
        current_hour = datetime.now().hour
        if 6 <= current_hour < 12:
            time_context = "It's morning - responses can include appropriate greetings."
        elif 12 <= current_hour < 17:
            time_context = "It's afternoon - maintain professional energy."
        elif 17 <= current_hour < 22:
            time_context = "It's evening - be mindful people may be wrapping up work."
        else:
            time_context = "It's late - keep responses brief and to the point."
        
        additional_instructions += f"\n{time_context}"
        
        # Format the complete prompt
        return cls.BASE_TEMPLATE.format(
            context_section=context_section,
            user_name=user_name,
            current_message=current_message,
            additional_instructions=additional_instructions
        )

    @classmethod
    def generate_thread_aware_prompt(cls, user_name: str, current_message: str,
                                   thread_parent: Dict, thread_messages: List[Dict],
                                   similar_messages: List[Dict] = None,
                                   knowledge_documents: List[Dict] = None) -> str:
        """Generate a prompt specifically for thread responses"""
        
        context_parts = []
        
        # Add knowledge documents if available
        if knowledge_documents:
            knowledge_context = "Relevant knowledge from uploaded documents:\n"
            for i, doc in enumerate(knowledge_documents[:2], 1):
                filename = doc.get('filename', 'Unknown')
                content = doc.get('content', '')
                chunk_index = doc.get('chunk_index', '0')
                similarity = doc.get('similarity_score', 0)
                
                # Include more content for thread context (but still limit for readability)
                if len(content) > 300:
                    content = content[:300] + "..."
                knowledge_context += f"{i}. From {filename} (chunk {chunk_index}, similarity: {similarity:.3f}):\n{content}\n\n"
            context_parts.append(knowledge_context.strip())
        
        # Add thread context
        thread_context = f"Original thread message from {thread_parent.get('user_name', 'Unknown')}:\n"
        thread_context += f"\"{thread_parent.get('text', '')}\"\n\n"
        thread_context += "Thread discussion:\n"
        
        for msg in thread_messages[-5:]:  # Last 5 thread messages
            user = msg.get('user_name', 'Unknown')
            text = msg.get('text', '')
            thread_context += f"{user}: {text}\n"
        context_parts.append(thread_context.strip())
        
        # Combine all context
        full_context = "\n\n".join(context_parts)
        
        prompt = f"""You are helping to suggest a response in a Slack thread.

{full_context}

Current message from {user_name}:
"{current_message}"

Generate a response that:
- Continues the thread discussion naturally
- Addresses the specific point raised
- Uses relevant knowledge from documents when applicable
- Maintains thread coherence
- Is concise and relevant to the thread topic

Response:"""
        
        return prompt