"""
Enhanced AI Integration System - Production Implementation

Integrates knowledge graph, multi-collection search, and document structure
to provide context-rich AI responses with source attribution.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json

from semantic_search_engine import EnhancedSearchSystem
from entity_relationship_graph import get_knowledge_graph
from agent.providers.provider_configuration_manager import get_provider_manager

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class AIContext:
    """Rich context for AI generation"""
    # Query information
    query: str
    intent: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    
    # Retrieved information
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_graph_context: Dict[str, Any] = field(default_factory=dict)
    hierarchical_context: Dict[str, Any] = field(default_factory=dict)
    
    # Conversation context
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    source_documents: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class EnhancedPrompt:
    """Enhanced prompt with structured components"""
    system_prompt: str
    user_prompt: str
    context_sections: Dict[str, str] = field(default_factory=dict)
    
    # Additional components
    examples: List[Dict[str, str]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    output_format: Optional[str] = None
    
    # Metadata
    total_tokens: int = 0
    provider_optimizations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIResponse:
    """Enhanced AI response with attribution"""
    content: str
    confidence: float
    
    # Attribution
    sources: List[Dict[str, Any]] = field(default_factory=list)
    entities_used: List[str] = field(default_factory=list)
    knowledge_graph_nodes: List[str] = field(default_factory=list)
    
    # Metadata
    provider: str = ""
    model: str = ""
    processing_time: float = 0.0
    tokens_used: Dict[str, int] = field(default_factory=dict)
    
    # Additional insights
    follow_up_questions: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)


class EnhancedAIIntegration:
    """
    Production-ready AI integration system that combines all enhanced features
    for intelligent, context-aware AI responses.
    """
    
    def __init__(self):
        # Core components
        self.search_system = None
        self.knowledge_graph = None
        self.provider_manager = None
        
        # Configuration
        self.config = {
            # Context retrieval
            "max_search_results": 10,
            "min_relevance_score": 0.6,
            "include_knowledge_graph": True,
            "knowledge_graph_depth": 2,
            
            # Prompt construction
            "max_context_length": 4000,
            "include_examples": True,
            "include_constraints": True,
            "entity_emphasis": True,
            
            # Response generation
            "default_temperature": 0.7,
            "default_max_tokens": 1000,
            "include_sources": True,
            "generate_follow_ups": True,
            
            # Provider optimization
            "optimize_for_provider": True,
            "fallback_enabled": True,
            
            # Performance
            "cache_prompts": True,
            "cache_ttl": 300
        }
        
        # Prompt templates
        self.templates = {
            "system": {
                "default": """You are an intelligent assistant with access to a comprehensive knowledge base. 
Provide accurate, helpful responses based on the provided context. Always cite your sources when possible.""",
                
                "technical": """You are a technical expert assistant. Provide detailed, accurate technical information 
based on the documentation and knowledge base provided. Include code examples when relevant.""",
                
                "conversational": """You are a friendly, conversational assistant. Provide helpful responses 
in a natural, engaging tone while remaining accurate and informative."""
            },
            
            "context_section": """## Relevant Information from Knowledge Base

{search_results}

## Related Concepts and Entities

{knowledge_graph}

## Document Structure Context

{hierarchical_context}
""",
            
            "source_attribution": """
Sources:
{sources}
""",
            
            "constraints": [
                "Base your response on the provided context",
                "Cite specific sources when making claims",
                "Acknowledge when information is not available",
                "Maintain consistency with previous responses"
            ]
        }
        
        # Cache
        self.prompt_cache = {}
        self.context_cache = {}
        
        logger.info("Enhanced AI Integration initialized")
    
    async def initialize(self):
        """Initialize AI integration components"""
        logger.info("Initializing Enhanced AI Integration")
        
        # Initialize components
        self.search_system = EnhancedSearchSystem()
        await self.search_system.initialize()
        
        self.knowledge_graph = get_knowledge_graph()
        self.provider_manager = get_provider_manager()
        
        logger.info("Enhanced AI Integration initialized successfully")
    
    async def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AIResponse:
        """
        Generate an AI response with full context enhancement
        
        Args:
            query: User query or prompt
            conversation_history: Previous conversation messages
            user_context: User-specific context (preferences, role, etc.)
            provider: Specific LLM provider to use
            model: Specific model to use
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            AIResponse with content, sources, and metadata
        """
        start_time = datetime.now()
        
        try:
            # Build AI context
            ai_context = await self._build_ai_context(
                query,
                conversation_history or [],
                user_context or {}
            )
            
            # Create enhanced prompt
            enhanced_prompt = await self._create_enhanced_prompt(
                ai_context,
                kwargs.get("prompt_style", "default")
            )
            
            # Optimize for provider if specified
            if provider and self.config["optimize_for_provider"]:
                enhanced_prompt = self._optimize_prompt_for_provider(
                    enhanced_prompt,
                    provider
                )
            
            # Generate response
            response_data = await self._generate_with_provider(
                enhanced_prompt,
                provider=provider,
                model=model,
                temperature=temperature or self.config["default_temperature"],
                max_tokens=max_tokens or self.config["default_max_tokens"],
                **kwargs
            )
            
            # Process and enhance response
            ai_response = await self._process_response(
                response_data,
                ai_context,
                enhanced_prompt
            )
            
            # Add processing time
            ai_response.processing_time = (datetime.now() - start_time).total_seconds()
            
            # Cache successful response
            if self.config["cache_prompts"]:
                cache_key = self._get_cache_key(query, conversation_history)
                self.prompt_cache[cache_key] = ai_response
            
            logger.info(f"Generated AI response in {ai_response.processing_time:.3f}s")
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            raise
    
    async def _build_ai_context(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        user_context: Dict[str, Any]
    ) -> AIContext:
        """Build comprehensive AI context"""
        context = AIContext(
            query=query,
            conversation_history=conversation_history,
            user_context=user_context
        )
        
        # Search for relevant information
        if self.config["max_search_results"] > 0:
            search_response = await self.search_system.search(
                query=query,
                search_type="hybrid",
                limit=self.config["max_search_results"],
                use_knowledge_graph=self.config["include_knowledge_graph"],
                min_relevance_score=self.config["min_relevance_score"]
            )
            
            # Extract search results
            context.search_results = [
                {
                    "content": result.content,
                    "source": result.document_title or result.document_id,
                    "relevance": result.score,
                    "metadata": result.metadata
                }
                for result in search_response.results
            ]
            
            # Extract entities and concepts
            context.entities = search_response.query.entities
            context.concepts = search_response.query.concepts
            context.intent = search_response.query.intent
            
            # Track source documents
            context.source_documents = list(set([
                result.document_id for result in search_response.results
            ]))
        
        # Get knowledge graph context
        if self.config["include_knowledge_graph"] and context.entities:
            context.knowledge_graph_context = await self._get_knowledge_graph_context(
                context.entities,
                depth=self.config["knowledge_graph_depth"]
            )
        
        # Build hierarchical context
        if context.source_documents:
            context.hierarchical_context = await self._build_hierarchical_context(
                context.source_documents,
                query
            )
        
        # Calculate confidence scores
        context.confidence_scores = self._calculate_context_confidence(context)
        
        return context
    
    async def _get_knowledge_graph_context(
        self,
        entities: List[str],
        depth: int = 2
    ) -> Dict[str, Any]:
        """Get relevant knowledge graph context"""
        kg_context = {
            "entities": {},
            "relationships": [],
            "clusters": []
        }
        
        # Get information for each entity
        for entity in entities[:5]:  # Limit for performance
            # Find related entities
            related = await self.knowledge_graph.find_related_entities(
                entity,
                max_hops=depth,
                limit=5
            )
            
            kg_context["entities"][entity] = [
                {
                    "text": e.text,
                    "type": e.entity_type,
                    "relevance": score
                }
                for e, score in related
            ]
        
        # Get concept clusters
        clusters = await self.knowledge_graph.find_concept_clusters(entities)
        kg_context["clusters"] = [
            {
                "name": cluster.name,
                "keywords": cluster.keywords[:5]
            }
            for cluster in clusters[:3]
        ]
        
        return kg_context
    
    async def _build_hierarchical_context(
        self,
        document_ids: List[str],
        query: str
    ) -> Dict[str, Any]:
        """Build hierarchical document context"""
        hierarchical_context = {}
        
        for doc_id in document_ids[:3]:  # Limit for context size
            # Simple document context (hierarchical builder not implemented)
            hierarchical_context[doc_id] = {
                "structure": {"title": f"Document {doc_id}", "sections": []},
                "relevant_sections": [],
                "context_path": [f"Document {doc_id}"]
            }
        
        return hierarchical_context
    
    def _calculate_context_confidence(self, context: AIContext) -> Dict[str, float]:
        """Calculate confidence scores for different context aspects"""
        scores = {}
        
        # Search result confidence
        if context.search_results:
            scores["search"] = sum(r["relevance"] for r in context.search_results) / len(context.search_results)
        else:
            scores["search"] = 0.0
        
        # Entity confidence
        scores["entities"] = min(1.0, len(context.entities) * 0.2)
        
        # Knowledge graph confidence
        if context.knowledge_graph_context.get("entities"):
            scores["knowledge_graph"] = 0.8
        else:
            scores["knowledge_graph"] = 0.0
        
        # Overall confidence
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores
    
    async def _create_enhanced_prompt(
        self,
        context: AIContext,
        prompt_style: str = "default"
    ) -> EnhancedPrompt:
        """Create enhanced prompt from context"""
        # Select system prompt template
        system_prompt = self.templates["system"].get(
            prompt_style,
            self.templates["system"]["default"]
        )
        
        # Build context sections
        context_sections = {}
        
        # Search results section
        if context.search_results:
            search_text = self._format_search_results(context.search_results)
            context_sections["search_results"] = search_text
        
        # Knowledge graph section
        if context.knowledge_graph_context:
            kg_text = self._format_knowledge_graph(context.knowledge_graph_context)
            context_sections["knowledge_graph"] = kg_text
        
        # Hierarchical context section
        if context.hierarchical_context:
            hier_text = self._format_hierarchical_context(context.hierarchical_context)
            context_sections["hierarchical_context"] = hier_text
        
        # Build user prompt
        user_prompt = self._build_user_prompt(context, context_sections)
        
        # Add examples if configured
        examples = []
        if self.config["include_examples"]:
            examples = self._get_relevant_examples(context)
        
        # Add constraints
        constraints = []
        if self.config["include_constraints"]:
            constraints = self.templates["constraints"].copy()
            
            # Add entity-specific constraints
            if context.entities and self.config["entity_emphasis"]:
                constraints.append(
                    f"Pay special attention to information about: {', '.join(context.entities[:3])}"
                )
        
        # Create enhanced prompt
        enhanced_prompt = EnhancedPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_sections=context_sections,
            examples=examples,
            constraints=constraints
        )
        
        # Calculate tokens (simplified)
        enhanced_prompt.total_tokens = self._estimate_tokens(enhanced_prompt)
        
        return enhanced_prompt
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for prompt"""
        formatted = []
        
        for i, result in enumerate(results[:5]):  # Limit to top 5
            formatted.append(f"""
### Source {i+1}: {result['source']}
Relevance: {result['relevance']:.2f}

{result['content'][:500]}...
""")
        
        return "\n".join(formatted)
    
    def _format_knowledge_graph(self, kg_context: Dict[str, Any]) -> str:
        """Format knowledge graph context for prompt"""
        formatted = []
        
        # Format entities
        if kg_context.get("entities"):
            formatted.append("### Related Entities:")
            for entity, related in kg_context["entities"].items():
                if related:
                    related_texts = [f"{r['text']} ({r['type']})" for r in related[:3]]
                    formatted.append(f"- **{entity}** is related to: {', '.join(related_texts)}")
        
        # Format clusters
        if kg_context.get("clusters"):
            formatted.append("\n### Concept Clusters:")
            for cluster in kg_context["clusters"]:
                formatted.append(f"- **{cluster['name']}**: {', '.join(cluster['keywords'])}")
        
        return "\n".join(formatted)
    
    def _format_hierarchical_context(self, hier_context: Dict[str, Any]) -> str:
        """Format hierarchical context for prompt"""
        formatted = []
        
        for doc_id, context in hier_context.items():
            if context.get("relevant_sections"):
                formatted.append(f"### Document Structure ({doc_id}):")
                for section in context["relevant_sections"][:3]:
                    formatted.append(f"- {section.get('title', 'Section')}: {section.get('summary', '')[:200]}...")
        
        return "\n".join(formatted)
    
    def _build_user_prompt(
        self,
        context: AIContext,
        context_sections: Dict[str, str]
    ) -> str:
        """Build complete user prompt"""
        parts = []
        
        # Add conversation history if present
        if context.conversation_history:
            parts.append("## Previous Conversation:")
            for msg in context.conversation_history[-3:]:  # Last 3 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("")
        
        # Add context sections
        if context_sections:
            context_text = self.templates["context_section"].format(**context_sections)
            parts.append(context_text)
        
        # Add the actual query
        parts.append(f"## Current Query:\n{context.query}")
        
        # Add user context if relevant
        if context.user_context:
            parts.append(f"\n## User Context:")
            for key, value in context.user_context.items():
                if key in ["role", "expertise", "preferences"]:
                    parts.append(f"- {key}: {value}")
        
        return "\n\n".join(parts)
    
    def _get_relevant_examples(self, context: AIContext) -> List[Dict[str, str]]:
        """Get relevant examples based on context"""
        examples = []
        
        # This is simplified - in production would retrieve from example database
        if context.intent == "technical_question":
            examples.append({
                "query": "How do I implement error handling?",
                "response": "Here's a comprehensive approach to error handling..."
            })
        
        return examples
    
    def _optimize_prompt_for_provider(
        self,
        prompt: EnhancedPrompt,
        provider: str
    ) -> EnhancedPrompt:
        """Optimize prompt for specific provider"""
        optimizations = {}
        
        if provider == "openai":
            # OpenAI optimizations
            optimizations["response_format"] = {"type": "text"}
            optimizations["system_message_style"] = "direct"
            
        elif provider == "anthropic":
            # Anthropic optimizations
            optimizations["xml_tags"] = True
            optimizations["thinking_style"] = "step-by-step"
            
        elif provider == "ollama":
            # Ollama optimizations (local models)
            optimizations["concise_context"] = True
            optimizations["max_context_length"] = 2000
        
        prompt.provider_optimizations = optimizations
        
        # Apply optimizations
        if optimizations.get("concise_context"):
            # Reduce context size for local models
            for key, value in prompt.context_sections.items():
                prompt.context_sections[key] = value[:optimizations.get("max_context_length", 4000)]
        
        return prompt
    
    async def _generate_with_provider(
        self,
        prompt: EnhancedPrompt,
        provider: Optional[str],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using LLM provider"""
        # Get active provider if not specified
        if not provider:
            active_provider = self.provider_manager.get_active_provider()
            if not active_provider:
                raise ValueError("No active LLM provider configured")
            provider = active_provider["name"]
        
        # Prepare messages
        messages = [
            {"role": "system", "content": prompt.system_prompt}
        ]
        
        # Add examples if present
        for example in prompt.examples:
            messages.append({"role": "user", "content": example["query"]})
            messages.append({"role": "assistant", "content": example["response"]})
        
        # Add main prompt
        if prompt.constraints:
            constraints_text = "\n\nPlease follow these guidelines:\n" + "\n".join(f"- {c}" for c in prompt.constraints)
            user_content = prompt.user_prompt + constraints_text
        else:
            user_content = prompt.user_prompt
        
        messages.append({"role": "user", "content": user_content})
        
        # Generate response
        try:
            response = await self.provider_manager.generate(
                provider_name=provider,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return {
                "content": response.get("content", ""),
                "provider": provider,
                "model": response.get("model", model or "default"),
                "usage": response.get("usage", {}),
                "raw_response": response
            }
            
        except Exception as e:
            logger.error(f"Provider {provider} failed: {e}")
            
            # Try fallback if enabled
            if self.config["fallback_enabled"]:
                return await self._fallback_generation(prompt, temperature, max_tokens)
            else:
                raise
    
    async def _fallback_generation(
        self,
        prompt: EnhancedPrompt,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Fallback generation using alternative provider"""
        # This is simplified - would try alternative providers
        return {
            "content": "I apologize, but I'm unable to generate a response at this time. Please try again later.",
            "provider": "fallback",
            "model": "none",
            "usage": {}
        }
    
    async def _process_response(
        self,
        response_data: Dict[str, Any],
        context: AIContext,
        prompt: EnhancedPrompt
    ) -> AIResponse:
        """Process and enhance the AI response"""
        ai_response = AIResponse(
            content=response_data["content"],
            confidence=context.confidence_scores.get("overall", 0.5),
            provider=response_data["provider"],
            model=response_data["model"],
            tokens_used=response_data.get("usage", {})
        )
        
        # Add sources if configured
        if self.config["include_sources"] and context.search_results:
            ai_response.sources = [
                {
                    "title": result["source"],
                    "relevance": result["relevance"],
                    "excerpt": result["content"][:200] + "..."
                }
                for result in context.search_results[:3]
            ]
        
        # Add entities used
        ai_response.entities_used = context.entities
        
        # Add knowledge graph nodes
        if context.knowledge_graph_context.get("entities"):
            ai_response.knowledge_graph_nodes = list(
                context.knowledge_graph_context["entities"].keys()
            )
        
        # Generate follow-up questions if configured
        if self.config["generate_follow_ups"]:
            ai_response.follow_up_questions = await self._generate_follow_ups(
                context,
                ai_response.content
            )
        
        # Extract related topics
        ai_response.related_topics = self._extract_related_topics(
            context,
            ai_response.content
        )
        
        return ai_response
    
    async def _generate_follow_ups(
        self,
        context: AIContext,
        response: str
    ) -> List[str]:
        """Generate follow-up questions"""
        follow_ups = []
        
        # Entity-based follow-ups
        for entity in context.entities[:2]:
            follow_ups.append(f"Tell me more about {entity}")
        
        # Concept-based follow-ups
        if context.concepts:
            follow_ups.append(f"How does this relate to {context.concepts[0]}?")
        
        # Intent-based follow-ups
        if context.intent == "technical_question":
            follow_ups.append("Can you provide a code example?")
        elif context.intent == "explanation":
            follow_ups.append("Can you explain this in simpler terms?")
        
        return follow_ups[:3]  # Limit to 3
    
    def _extract_related_topics(
        self,
        context: AIContext,
        response: str
    ) -> List[str]:
        """Extract related topics from context and response"""
        topics = set()
        
        # Add concept clusters
        if context.knowledge_graph_context.get("clusters"):
            for cluster in context.knowledge_graph_context["clusters"]:
                topics.add(cluster["name"])
        
        # Add high-relevance entities
        for entity in context.entities[:3]:
            topics.add(entity)
        
        return list(topics)[:5]
    
    def _estimate_tokens(self, prompt: EnhancedPrompt) -> int:
        """Estimate token count for prompt"""
        # Simplified estimation (4 characters per token)
        total_chars = len(prompt.system_prompt) + len(prompt.user_prompt)
        
        for section in prompt.context_sections.values():
            total_chars += len(section)
        
        for example in prompt.examples:
            total_chars += len(example.get("query", "")) + len(example.get("response", ""))
        
        return total_chars // 4
    
    def _get_cache_key(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> str:
        """Generate cache key for prompt"""
        history_str = ""
        if conversation_history:
            history_str = "|".join([
                f"{msg.get('role', '')}:{msg.get('content', '')[:50]}"
                for msg in conversation_history[-3:]
            ])
        
        return f"{query}|{history_str}"


# Global instance
_ai_integration = None


def get_enhanced_ai_integration() -> EnhancedAIIntegration:
    """Get global enhanced AI integration instance"""
    global _ai_integration
    if _ai_integration is None:
        _ai_integration = EnhancedAIIntegration()
    return _ai_integration


# Test function
if __name__ == "__main__":
    async def test_ai_integration():
        ai_system = EnhancedAIIntegration()
        await ai_system.initialize()
        
        # Test response generation
        response = await ai_system.generate_response(
            query="What are the best practices for error handling in Python?",
            conversation_history=[
                {"role": "user", "content": "I'm working on a Python project"},
                {"role": "assistant", "content": "I'd be happy to help with your Python project!"}
            ],
            user_context={"role": "developer", "expertise": "intermediate"},
            temperature=0.7,
            max_tokens=500
        )
        
        print(f"AI Response:")
        print(f"Content: {response.content[:200]}...")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Provider: {response.provider}")
        print(f"Sources: {len(response.sources)}")
        print(f"Entities Used: {response.entities_used}")
        print(f"Processing Time: {response.processing_time:.3f}s")
        
        if response.follow_up_questions:
            print(f"\nFollow-up Questions:")
            for q in response.follow_up_questions:
                print(f"- {q}")
    
    # Run test
    asyncio.run(test_ai_integration())