"""
Enhanced suggestion service with conversation history and thread awareness
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

from conversation_prompt_templates import PromptTemplates
from agent.providers.provider_configuration_manager import get_provider_manager

logger = logging.getLogger(__name__)


class SuggestionService:
    """Enhanced suggestion service with context awareness"""
    
    def __init__(self, vector_db, provider_manager=None):
        self.vector_db = vector_db  # This is now EnhancedSearchSystem
        # Use the new provider manager system
        self.provider_manager = provider_manager or get_provider_manager()
        self.prompt_templates = PromptTemplates()
    
    async def get_conversation_history(self, conversation_id: str, 
                                     limit: int = 10,
                                     time_window_hours: int = 24) -> List[Dict]:
        """Retrieve recent conversation history"""
        if not conversation_id:
            return []
        
        try:
            # Search for recent messages in this conversation using EnhancedSearchSystem
            results = await self.vector_db.search_multiple_collections(
                query="*",  # Get all messages
                collections=["slack_messages"],
                filters={"channel_id": conversation_id},
                limit=limit
            )
            
            messages = []
            for result in results:
                metadata = result.get('metadata', {})
                messages.append({
                    'text': result.get('content', ''),
                    'user_name': metadata.get('user_id', 'Unknown'),
                    'user_id': metadata.get('user_id', ''),
                    'timestamp': metadata.get('timestamp', ''),
                    'created_at': metadata.get('timestamp', ''),
                    'thread_ts': metadata.get('thread_id', '')
                })
            
            # Sort by timestamp (newest first) and limit
            messages.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return messages[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    async def get_thread_messages(self, conversation_id: str, 
                                thread_ts: str,
                                limit: int = 10) -> List[Dict]:
        """Retrieve messages from a specific thread"""
        try:
            # Search for messages in this thread using EnhancedSearchSystem
            results = await self.vector_db.search_multiple_collections(
                query="*",
                collections=["slack_messages"],
                filters={
                    "channel_id": conversation_id,
                    "thread_id": thread_ts
                },
                limit=limit
            )
            
            messages = []
            for result in results:
                metadata = result.get('metadata', {})
                messages.append({
                    'text': result.get('content', ''),
                    'user_name': metadata.get('user_id', 'Unknown'),
                    'user_id': metadata.get('user_id', ''),
                    'timestamp': metadata.get('timestamp', ''),
                    'created_at': metadata.get('timestamp', '')
                })
            
            # Sort by timestamp (oldest first for thread continuity)
            messages.sort(key=lambda x: x.get('timestamp', ''))
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving thread messages: {e}")
            return []
    
    def _extract_message_text(self, result: Dict) -> str:
        """Extract clean message text from search result"""
        full_text = result.get('text', '')
        if '| Message: ' in full_text:
            message_text = full_text.split('| Message: ', 1)[1]
            if '(in thread)' in message_text:
                message_text = message_text.replace(' (in thread)', '')
        else:
            message_text = full_text
        return message_text.strip()
    
    async def generate_enhanced_suggestion(self, 
                                         user_name: str,
                                         user_id: str,
                                         content: str,
                                         conversation_id: Optional[str] = None,
                                         thread_ts: Optional[str] = None) -> Dict[str, Any]:
        """Generate enhanced suggestion with full context awareness"""
        
        start_time = datetime.now()
        
        try:
            # Step 1: Vector search for similar messages
            logger.info(f"Searching for messages similar to: '{content}'")
            logger.info(f"Conversation ID filter: {conversation_id}")
            
            # First try with no filters to see if we get any results
            similar_results = await self.vector_db.search_multiple_collections(
                query=content,
                collections=["slack_messages"],
                limit=5,
                filters=None  # Remove channel filter temporarily to debug
            )
            
            logger.info(f"Found {len(similar_results)} similar messages without filters")
            
            # Process similar messages
            similar_messages = []
            for result in similar_results:
                metadata = result.get('metadata', {})
                similar_messages.append({
                    'text': result.get('content', ''),
                    'user_name': metadata.get('user_name', 'Unknown'),
                    'similarity_score': result.get('score', 0)
                })
            
            # Step 1.5: Search knowledge documents for relevant information
            logger.info(f"Searching knowledge documents for: '{content}'")
            document_results = []
            
            # Get available collections first
            try:
                # Use the client directly since get_collections method doesn't exist
                available_collections = self.vector_db.client.list_collections()
                collection_names = [col.name for col in available_collections]
            except Exception as e:
                logger.warning(f"Could not get available collections: {e}")
                collection_names = []
            
            # Try to find document collections in order of preference
            document_collections_to_try = [
                "documents", 
                "slack_documents", 
                "uploaded_documents",
                "docling_documents",
                "document_chunks"
            ]
            
            for collection_name in document_collections_to_try:
                if collection_name in collection_names:
                    try:
                        document_results = await self.vector_db.search_multiple_collections(
                            query=content,
                            collections=[collection_name],
                            limit=3
                        )
                        logger.info(f"Found {len(document_results)} document results in {collection_name}")
                        break
                    except Exception as e:
                        logger.warning(f"Error searching {collection_name}: {e}")
                        continue
            
            if not document_results:
                logger.info("No document collections available or no results found, continuing without documents")
            
            # Process document results
            relevant_documents = []
            for result in document_results:
                metadata = result.get('metadata', {})
                relevant_documents.append({
                    'content': result.get('content', ''),
                    'filename': metadata.get('filename', 'Unknown'),
                    'chunk_index': metadata.get('chunk_index', '0'),
                    'similarity_score': result.get('score', 0),
                    'file_type': metadata.get('file_type', 'unknown')
                })
            
            logger.info(f"Found {len(relevant_documents)} relevant document chunks")
            
            # Log detailed information about retrieved document chunks
            if relevant_documents:
                logger.info("=== RETRIEVED KNOWLEDGE BASE CHUNKS ===")
                for i, doc in enumerate(relevant_documents, 1):
                    logger.info(f"Chunk {i}: {doc['filename']} (chunk {doc['chunk_index']}) - Similarity: {doc['similarity_score']:.3f}")
                    logger.info(f"  File type: {doc['file_type']}")
                    logger.info(f"  FULL CONTENT:")
                    logger.info(f"  {doc['content']}")
                    logger.info(f"  --- END OF CHUNK {i} CONTENT ---")
                logger.info("=== END KNOWLEDGE BASE CHUNKS ===")
            else:
                logger.info("No relevant document chunks found in knowledge base")
            
            # Step 2: Get conversation history if available
            conversation_history = []
            if conversation_id:
                conversation_history = await self.get_conversation_history(
                    conversation_id, 
                    limit=10,
                    time_window_hours=24
                )
            
            # Step 3: Get thread context if in a thread
            thread_messages = []
            thread_parent = None
            if thread_ts and conversation_id:
                thread_messages = await self.get_thread_messages(
                    conversation_id,
                    thread_ts,
                    limit=10
                )
                # First message in thread is usually the parent
                if thread_messages:
                    thread_parent = thread_messages[0]
            
            # Step 4: Check for active LLM provider
            active_provider_info = self.provider_manager.get_active_provider()
            
            # Debug logging - print to console for immediate visibility
            print(f"DEBUG: Similar messages found: {len(similar_messages)}")
            print(f"DEBUG: Conversation history: {len(conversation_history)}")
            print(f"DEBUG: Relevant documents: {len(relevant_documents)}")
            print(f"DEBUG: Active provider exists: {bool(active_provider_info)}")
            
            # Show first few similar messages for debugging
            if similar_messages:
                print(f"DEBUG: First similar message: {similar_messages[0]}")
            
            # Temporarily bypass provider check for debugging
            if active_provider_info and (similar_messages or conversation_history or relevant_documents):
                try:
                    # Detect conversation type
                    conversation_type = PromptTemplates.detect_conversation_type(
                        content,
                        [msg['text'] for msg in similar_messages[:3]]
                    )
                    
                    # Generate appropriate prompt with knowledge documents
                    if relevant_documents:
                        logger.info(f"Including {len(relevant_documents)} knowledge document chunks in LLM prompt")
                    
                    if thread_parent:
                        # Thread-specific prompt with documents
                        prompt = PromptTemplates.generate_thread_aware_prompt(
                            user_name=user_name,
                            current_message=content,
                            thread_parent=thread_parent,
                            thread_messages=thread_messages[1:],  # Exclude parent
                            similar_messages=similar_messages,
                            knowledge_documents=relevant_documents
                        )
                        logger.info("Generated thread-aware prompt with knowledge base integration")
                    else:
                        # Regular conversation prompt with documents
                        prompt = PromptTemplates.generate_prompt(
                            user_name=user_name,
                            current_message=content,
                            similar_messages=similar_messages,
                            conversation_history=conversation_history,
                            thread_messages=None,
                            conversation_type=conversation_type,
                            knowledge_documents=relevant_documents
                        )
                        logger.info("Generated standard prompt with knowledge base integration")
                    
                    # Log the complete prompt being sent to LLM for debugging
                    logger.info("=== COMPLETE LLM PROMPT ===")
                    logger.info(f"Prompt length: {len(prompt)} characters")
                    logger.info("Full prompt content:")
                    logger.info(prompt)
                    logger.info("=== END LLM PROMPT ===")
                    
                    # Get LLM response using new provider manager
                    llm_start = datetime.now()
                    llm_response = await self.provider_manager.generate(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=150
                    )
                    llm_time = (datetime.now() - llm_start).total_seconds()
                    
                    # Log the raw LLM response
                    logger.info("=== RAW LLM RESPONSE ===")
                    logger.info(f"Response time: {llm_time:.2f} seconds")
                    logger.info(f"Raw response: {llm_response}")
                    logger.info("=== END RAW LLM RESPONSE ===")
                    
                    suggestion_text = llm_response.get('content', llm_response.get('text', ''))
                    
                    # Clean up the suggestion
                    suggestion_text = suggestion_text.strip()
                    if suggestion_text.startswith('"') and suggestion_text.endswith('"'):
                        suggestion_text = suggestion_text[1:-1]
                    
                    logger.info(f"Final cleaned suggestion: {suggestion_text}")
                    
                    total_time = (datetime.now() - start_time).total_seconds()
                    
                    return {
                        "suggestion": suggestion_text,
                        "metadata": {
                            "source": f"Enhanced AI ({active_provider_info['name']})",
                            "conversation_type": conversation_type,
                            "context_used": {
                                "similar_messages": len(similar_messages),
                                "conversation_history": len(conversation_history),
                                "thread_messages": len(thread_messages),
                                "knowledge_documents": len(relevant_documents),
                                "is_thread_response": bool(thread_parent)
                            },
                            "performance": {
                                "total_time": round(total_time, 2),
                                "llm_time": round(llm_time, 2),
                                "vector_search_time": round(total_time - llm_time, 2)
                            }
                        },
                        "fallback_suggestions": [msg['text'] for msg in similar_messages[:3]]
                    }
                    
                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    # Fall through to vector-only response
            
            # Fallback: Vector search only response
            if similar_messages or relevant_documents:
                # Format top similar messages as suggestions
                suggestions = []
                for msg in similar_messages[:3]:
                    suggestions.append(msg['text'])
                
                # If we have document results but no similar messages, create suggestion from documents
                if not suggestions and relevant_documents:
                    for doc in relevant_documents[:2]:
                        # Create a helpful response based on document content
                        doc_snippet = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
                        suggestions.append(f"Based on {doc['filename']}: {doc_snippet}")
                
                total_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "suggestion": suggestions[0] if suggestions else "I'd be happy to help with that!",
                    "metadata": {
                        "source": "Vector Search" + (" + Knowledge Base" if relevant_documents else ""),
                        "conversation_type": "unknown",
                        "context_used": {
                            "similar_messages": len(similar_messages),
                            "conversation_history": 0,
                            "thread_messages": 0,
                            "knowledge_documents": len(relevant_documents),
                            "is_thread_response": False
                        },
                        "performance": {
                            "total_time": round(total_time, 2),
                            "llm_time": 0,
                            "vector_search_time": round(total_time, 2)
                        }
                    },
                    "fallback_suggestions": suggestions
                }
            else:
                # No context available - return generic suggestion
                total_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "suggestion": "Thanks for your message! How can I help you today?",
                    "metadata": {
                        "source": "Default",
                        "conversation_type": "unknown", 
                        "context_used": {
                            "similar_messages": 0,
                            "conversation_history": 0,
                            "thread_messages": 0,
                            "knowledge_documents": 0,
                            "is_thread_response": False
                        },
                        "performance": {
                            "total_time": round(total_time, 2),
                            "llm_time": 0,
                            "vector_search_time": 0
                        }
                    },
                    "fallback_suggestions": []
                }
                
        except Exception as e:
            logger.error(f"Error generating enhanced suggestion: {e}")
            return {
                "suggestion": "I'd be happy to help with that!",
                "metadata": {
                    "source": "Error Fallback",
                    "error": str(e)
                },
                "fallback_suggestions": []
            }
    
    def _is_provider_configured(self, provider_info: Dict[str, Any]) -> bool:
        """Check if a provider is properly configured"""
        try:
            provider = provider_info.get("provider")
            if not provider:
                return False
            
            # Check if provider has required attributes for the provider type
            provider_name = provider_info.get("name", "").lower()
            
            if provider_name in ["openai", "anthropic", "openrouter"]:
                # These providers need an API key
                return hasattr(provider, 'api_key') and provider.api_key
            elif provider_name == "ollama":
                # Ollama just needs a base_url
                return hasattr(provider, 'base_url') and provider.base_url
            else:
                # Unknown provider, assume it's configured if it exists
                return True
                
        except Exception as e:
            logger.error(f"Error checking provider configuration: {e}")
            return False