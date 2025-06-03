"""
Phase 3: Enhanced Prompt Generator

Advanced prompt generation system that leverages hierarchical document structure,
entities, relationships, and contextual understanding for superior AI interactions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re
from collections import defaultdict

from hierarchy_manager import (
    HierarchicalContextManager, 
    ContextRequest, 
    ContextType, 
    HierarchyLevel,
    get_hierarchical_context_manager
)

# Set up logging
logger = logging.getLogger(__name__)

class PromptType(Enum):
    """Types of prompts that can be generated"""
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    EXPLANATION = "explanation"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    CREATIVE_WRITING = "creative_writing"
    CODE_GENERATION = "code_generation"
    PROBLEM_SOLVING = "problem_solving"

class PromptComplexity(Enum):
    """Complexity levels for prompt generation"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

class ContextIntegrationStrategy(Enum):
    """Strategies for integrating context into prompts"""
    HIERARCHICAL = "hierarchical"
    THEMATIC = "thematic"
    ENTITY_FOCUSED = "entity_focused"
    RELATIONSHIP_BASED = "relationship_based"
    ADAPTIVE = "adaptive"

@dataclass
class PromptTemplate:
    """Template for prompt generation"""
    template_id: str
    prompt_type: PromptType
    complexity: PromptComplexity
    base_template: str
    context_placeholders: List[str]
    entity_placeholders: List[str]
    relationship_placeholders: List[str]
    metadata_placeholders: List[str]
    
    # Template configuration
    max_context_length: int = 3000
    preferred_context_types: List[ContextType] = field(default_factory=list)
    integration_strategy: ContextIntegrationStrategy = ContextIntegrationStrategy.ADAPTIVE
    
    # Quality parameters
    requires_entities: bool = False
    requires_relationships: bool = False
    requires_hierarchy: bool = False
    min_context_elements: int = 1

@dataclass
class PromptRequest:
    """Request for enhanced prompt generation"""
    user_query: str
    prompt_type: PromptType
    complexity: PromptComplexity = PromptComplexity.MODERATE
    
    # Context preferences
    max_context_length: int = 4000
    integration_strategy: ContextIntegrationStrategy = ContextIntegrationStrategy.ADAPTIVE
    include_entities: bool = True
    include_relationships: bool = True
    include_metadata: bool = True
    
    # User context
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    user_expertise_level: str = "intermediate"
    
    # Output preferences
    preferred_style: str = "professional"
    output_format: str = "detailed"
    language: str = "en"

@dataclass
class EnhancedPrompt:
    """Generated enhanced prompt with metadata"""
    prompt_text: str
    prompt_type: PromptType
    complexity: PromptComplexity
    
    # Context information
    context_summary: str
    entities_used: List[Dict[str, Any]]
    relationships_used: List[Dict[str, Any]]
    hierarchy_info: Dict[str, Any]
    
    # Quality metrics
    context_coverage_score: float
    entity_integration_score: float
    relationship_integration_score: float
    overall_quality_score: float
    
    # Metadata
    template_used: str
    generation_timestamp: datetime
    context_sources: List[str]
    estimated_response_length: int

class EnhancedPromptGenerator:
    """
    Advanced prompt generator with hierarchical context integration
    """
    
    def __init__(self):
        self.context_manager = None  # Will be initialized lazily
        self.prompt_templates = {}
        self.prompt_history = []
        
        # Configuration
        self.config = {
            "max_prompt_length": 8000,
            "context_weight": 0.6,
            "entity_weight": 0.2,
            "relationship_weight": 0.2,
            "enable_adaptive_complexity": True,
            "enable_style_adaptation": True,
            "enable_prompt_optimization": True
        }
        
        # Initialize templates
        self._initialize_prompt_templates()
        
        logger.info("Enhanced Prompt Generator initialized")
    
    async def initialize(self):
        """Initialize the prompt generator"""
        if self.context_manager is None:
            self.context_manager = await get_hierarchical_context_manager()
    
    def _initialize_prompt_templates(self):
        """Initialize prompt templates for different types and complexities"""
        
        # Question Answering Templates
        self.prompt_templates["qa_simple"] = PromptTemplate(
            template_id="qa_simple",
            prompt_type=PromptType.QUESTION_ANSWERING,
            complexity=PromptComplexity.SIMPLE,
            base_template="""Based on the following context, please answer this question: {user_query}

Context:
{hierarchical_context}

Answer the question directly and concisely using the information provided.""",
            context_placeholders=["hierarchical_context"],
            entity_placeholders=["key_entities"],
            relationship_placeholders=[],
            metadata_placeholders=["context_sources"]
        )
        
        self.prompt_templates["qa_complex"] = PromptTemplate(
            template_id="qa_complex",
            prompt_type=PromptType.QUESTION_ANSWERING,
            complexity=PromptComplexity.COMPLEX,
            base_template="""You are an expert analyst with access to comprehensive documentation. Using the hierarchical context below, provide a detailed answer to this question: {user_query}

## Document Structure & Context:
{hierarchical_context}

## Key Entities & Concepts:
{entity_context}

## Relationships & Dependencies:
{relationship_context}

## Additional Metadata:
{metadata_context}

Please provide a comprehensive answer that:
1. Directly addresses the question using evidence from the context
2. Explains relevant relationships and dependencies
3. References specific sections or documents when appropriate
4. Considers multiple perspectives if applicable
5. Highlights any limitations or assumptions

Your response should be well-structured, thorough, and demonstrate deep understanding of the interconnected concepts.""",
            context_placeholders=["hierarchical_context"],
            entity_placeholders=["entity_context"],
            relationship_placeholders=["relationship_context"],
            metadata_placeholders=["metadata_context", "context_sources"],
            requires_entities=True,
            requires_relationships=True,
            requires_hierarchy=True
        )
        
        # Explanation Templates
        self.prompt_templates["explanation_moderate"] = PromptTemplate(
            template_id="explanation_moderate",
            prompt_type=PromptType.EXPLANATION,
            complexity=PromptComplexity.MODERATE,
            base_template="""Please explain the following topic in detail: {user_query}

Use this contextual information to provide a comprehensive explanation:

{hierarchical_context}

{entity_context}

Structure your explanation to:
- Start with a clear definition or overview
- Break down complex concepts into understandable parts
- Use examples from the context when relevant
- Explain how different concepts relate to each other
- Conclude with practical implications or applications""",
            context_placeholders=["hierarchical_context"],
            entity_placeholders=["entity_context"],
            relationship_placeholders=[],
            metadata_placeholders=["context_sources"]
        )
        
        # Analysis Templates
        self.prompt_templates["analysis_expert"] = PromptTemplate(
            template_id="analysis_expert",
            prompt_type=PromptType.ANALYSIS,
            complexity=PromptComplexity.EXPERT,
            base_template="""Conduct a comprehensive analysis of: {user_query}

## Hierarchical Information Structure:
{hierarchical_context}

## Entity Network & Key Concepts:
{entity_context}

## Relationship Matrix & Dependencies:
{relationship_context}

## Document Metadata & Sources:
{metadata_context}

Perform a multi-dimensional analysis that includes:

### 1. Structural Analysis
- Examine the hierarchical organization of information
- Identify patterns and structures in the data
- Assess completeness and coherence

### 2. Conceptual Analysis  
- Analyze key entities and their roles
- Examine concept relationships and dependencies
- Identify central themes and peripheral topics

### 3. Relationship Analysis
- Map interconnections between concepts
- Identify causal relationships and dependencies
- Analyze information flow and influence patterns

### 4. Critical Assessment
- Evaluate strengths and limitations
- Identify gaps or inconsistencies
- Assess reliability and validity of sources

### 5. Strategic Implications
- Draw actionable insights
- Recommend next steps or considerations
- Identify opportunities and risks

Support all conclusions with specific evidence from the provided context.""",
            context_placeholders=["hierarchical_context"],
            entity_placeholders=["entity_context"],
            relationship_placeholders=["relationship_context"],
            metadata_placeholders=["metadata_context", "context_sources"],
            requires_entities=True,
            requires_relationships=True,
            requires_hierarchy=True,
            min_context_elements=5
        )
        
        # Summarization Templates
        self.prompt_templates["summary_simple"] = PromptTemplate(
            template_id="summary_simple",
            prompt_type=PromptType.SUMMARIZATION,
            complexity=PromptComplexity.SIMPLE,
            base_template="""Please provide a concise summary of the following information related to: {user_query}

Content to summarize:
{hierarchical_context}

Create a clear, well-organized summary that captures the main points and key insights.""",
            context_placeholders=["hierarchical_context"],
            entity_placeholders=[],
            relationship_placeholders=[],
            metadata_placeholders=[]
        )
        
        # Comparison Templates
        self.prompt_templates["comparison_moderate"] = PromptTemplate(
            template_id="comparison_moderate",
            prompt_type=PromptType.COMPARISON,
            complexity=PromptComplexity.MODERATE,
            base_template="""Compare and contrast the following topic: {user_query}

Information for comparison:
{hierarchical_context}

Key entities and concepts:
{entity_context}

Create a structured comparison that:
- Identifies similarities and differences
- Organizes information into clear categories
- Uses specific examples from the context
- Provides balanced analysis
- Concludes with insights about the comparison""",
            context_placeholders=["hierarchical_context"],
            entity_placeholders=["entity_context"],
            relationship_placeholders=[],
            metadata_placeholders=["context_sources"]
        )
        
        logger.info(f"Initialized {len(self.prompt_templates)} prompt templates")
    
    async def generate_enhanced_prompt(self, request: PromptRequest) -> EnhancedPrompt:
        """Generate enhanced prompt with hierarchical context integration"""
        
        try:
            await self.initialize()
            
            # Select appropriate template
            template = self._select_template(request)
            
            # Get hierarchical context
            context_request = ContextRequest(
                query=request.user_query,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                max_context_length=request.max_context_length,
                hierarchy_strategy=request.integration_strategy.value,
                include_relationships=request.include_relationships,
                include_entities=request.include_entities
            )
            
            hierarchical_context = await self.context_manager.get_hierarchical_context(context_request)
            
            # Build context components
            context_components = await self._build_context_components(
                hierarchical_context, request, template
            )
            
            # Generate the prompt
            prompt_text = await self._generate_prompt_text(
                template, request, context_components
            )
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                prompt_text, hierarchical_context, context_components
            )
            
            # Create enhanced prompt
            enhanced_prompt = EnhancedPrompt(
                prompt_text=prompt_text,
                prompt_type=request.prompt_type,
                complexity=request.complexity,
                context_summary=context_components.get("summary", ""),
                entities_used=context_components.get("entities", []),
                relationships_used=context_components.get("relationships", []),
                hierarchy_info=hierarchical_context.get("metadata", {}),
                context_coverage_score=quality_metrics["context_coverage"],
                entity_integration_score=quality_metrics["entity_integration"],
                relationship_integration_score=quality_metrics["relationship_integration"],
                overall_quality_score=quality_metrics["overall_quality"],
                template_used=template.template_id,
                generation_timestamp=datetime.now(),
                context_sources=hierarchical_context.get("metadata", {}).get("sources", []),
                estimated_response_length=self._estimate_response_length(prompt_text, request.complexity)
            )
            
            # Store in history
            self.prompt_history.append(enhanced_prompt)
            
            logger.info(f"Generated enhanced prompt: {len(prompt_text)} chars, quality: {quality_metrics['overall_quality']:.2f}")
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced prompt: {e}")
            # Return basic prompt as fallback
            return EnhancedPrompt(
                prompt_text=f"Please answer the following question: {request.user_query}",
                prompt_type=request.prompt_type,
                complexity=request.complexity,
                context_summary="",
                entities_used=[],
                relationships_used=[],
                hierarchy_info={},
                context_coverage_score=0.0,
                entity_integration_score=0.0,
                relationship_integration_score=0.0,
                overall_quality_score=0.2,
                template_used="fallback",
                generation_timestamp=datetime.now(),
                context_sources=[],
                estimated_response_length=100
            )
    
    def _select_template(self, request: PromptRequest) -> PromptTemplate:
        """Select appropriate template based on request parameters"""
        
        # Build template key
        template_key = f"{request.prompt_type.value}_{request.complexity.value}"
        
        # Try exact match first
        if template_key in self.prompt_templates:
            return self.prompt_templates[template_key]
        
        # Find best matching template
        best_match = None
        best_score = 0
        
        for template in self.prompt_templates.values():
            score = 0
            
            # Match prompt type (highest priority)
            if template.prompt_type == request.prompt_type:
                score += 10
            
            # Match complexity
            complexity_scores = {
                PromptComplexity.SIMPLE: 1,
                PromptComplexity.MODERATE: 2,
                PromptComplexity.COMPLEX: 3,
                PromptComplexity.EXPERT: 4
            }
            
            req_complexity_score = complexity_scores.get(request.complexity, 2)
            template_complexity_score = complexity_scores.get(template.complexity, 2)
            
            # Prefer exact match, but allow close matches
            complexity_diff = abs(req_complexity_score - template_complexity_score)
            if complexity_diff == 0:
                score += 5
            elif complexity_diff == 1:
                score += 3
            elif complexity_diff == 2:
                score += 1
            
            if score > best_score:
                best_score = score
                best_match = template
        
        if best_match is None:
            # Fallback to first available template
            best_match = next(iter(self.prompt_templates.values()))
        
        logger.debug(f"Selected template: {best_match.template_id} (score: {best_score})")
        
        return best_match
    
    async def _build_context_components(
        self, 
        hierarchical_context: Dict[str, Any], 
        request: PromptRequest, 
        template: PromptTemplate
    ) -> Dict[str, Any]:
        """Build context components for prompt generation"""
        
        components = {}
        
        # Main hierarchical context
        components["hierarchical_context"] = hierarchical_context.get("context", "")
        
        # Entity context
        if request.include_entities and template.entity_placeholders:
            components["entity_context"] = await self._build_entity_context(
                hierarchical_context.get("metadata", {})
            )
            components["entities"] = hierarchical_context.get("metadata", {}).get("entities", [])
        
        # Relationship context
        if request.include_relationships and template.relationship_placeholders:
            components["relationship_context"] = await self._build_relationship_context(
                hierarchical_context.get("metadata", {})
            )
            components["relationships"] = hierarchical_context.get("metadata", {}).get("relationships", [])
        
        # Metadata context
        if request.include_metadata and template.metadata_placeholders:
            components["metadata_context"] = await self._build_metadata_context(
                hierarchical_context.get("metadata", {})
            )
        
        # Context summary
        components["summary"] = await self._build_context_summary(hierarchical_context)
        
        return components
    
    async def _build_entity_context(self, metadata: Dict[str, Any]) -> str:
        """Build entity context section"""
        
        entities_info = metadata.get("entities", {})
        if not entities_info:
            return "No specific entities identified in the context."
        
        entity_lines = []
        
        # Total entities
        if "total_entities" in entities_info:
            entity_lines.append(f"Total entities found: {entities_info['total_entities']}")
        
        # Entity types
        if "entity_types" in entities_info:
            entity_lines.append("Entity types:")
            for entity_type, count in entities_info["entity_types"].items():
                entity_lines.append(f"  - {entity_type}: {count}")
        
        # Unique entities
        if "unique_entities" in entities_info:
            entity_lines.append(f"Unique entities: {entities_info['unique_entities']}")
        
        return "\n".join(entity_lines)
    
    async def _build_relationship_context(self, metadata: Dict[str, Any]) -> str:
        """Build relationship context section"""
        
        relationships_info = metadata.get("relationships", {})
        if not relationships_info:
            return "No explicit relationships identified in the context."
        
        relationship_lines = []
        
        # Relationship counts
        for rel_type, count in relationships_info.items():
            if isinstance(count, int):
                relationship_lines.append(f"{rel_type.replace('_', ' ').title()}: {count}")
        
        if not relationship_lines:
            return "Relationship information is available but not detailed."
        
        return "\n".join(relationship_lines)
    
    async def _build_metadata_context(self, metadata: Dict[str, Any]) -> str:
        """Build metadata context section"""
        
        metadata_lines = []
        
        # Context statistics
        if "total_elements" in metadata:
            metadata_lines.append(f"Context elements: {metadata['total_elements']}")
        
        if "total_length" in metadata:
            metadata_lines.append(f"Total content length: {metadata['total_length']} characters")
        
        # Hierarchy levels
        if "hierarchy_levels" in metadata:
            levels = [level.replace("_", " ").title() for level in metadata["hierarchy_levels"]]
            metadata_lines.append(f"Hierarchy levels: {', '.join(levels)}")
        
        # Context types
        if "context_types" in metadata:
            types = [ctype.replace("_", " ").title() for ctype in metadata["context_types"]]
            metadata_lines.append(f"Content types: {', '.join(types)}")
        
        return "\n".join(metadata_lines) if metadata_lines else "Additional metadata available."
    
    async def _build_context_summary(self, hierarchical_context: Dict[str, Any]) -> str:
        """Build a summary of the context"""
        
        metadata = hierarchical_context.get("metadata", {})
        context_text = hierarchical_context.get("context", "")
        
        summary_parts = []
        
        # Basic statistics
        if metadata.get("total_elements"):
            summary_parts.append(f"{metadata['total_elements']} context elements")
        
        if metadata.get("total_length"):
            summary_parts.append(f"{metadata['total_length']} characters of content")
        
        # Content overview
        if context_text:
            # Extract first meaningful sentence as overview
            sentences = re.split(r'[.!?]\s+', context_text[:500])
            if sentences:
                summary_parts.append(f"Overview: {sentences[0]}")
        
        return "; ".join(summary_parts) if summary_parts else "Context information available"
    
    async def _generate_prompt_text(
        self, 
        template: PromptTemplate, 
        request: PromptRequest, 
        components: Dict[str, Any]
    ) -> str:
        """Generate the final prompt text"""
        
        try:
            # Start with base template
            prompt_text = template.base_template
            
            # Replace user query
            prompt_text = prompt_text.replace("{user_query}", request.user_query)
            
            # Replace context placeholders
            for placeholder in template.context_placeholders:
                if placeholder in components:
                    prompt_text = prompt_text.replace(f"{{{placeholder}}}", components[placeholder])
                else:
                    prompt_text = prompt_text.replace(f"{{{placeholder}}}", "")
            
            # Replace entity placeholders
            for placeholder in template.entity_placeholders:
                if placeholder in components:
                    prompt_text = prompt_text.replace(f"{{{placeholder}}}", components[placeholder])
                else:
                    prompt_text = prompt_text.replace(f"{{{placeholder}}}", "")
            
            # Replace relationship placeholders
            for placeholder in template.relationship_placeholders:
                if placeholder in components:
                    prompt_text = prompt_text.replace(f"{{{placeholder}}}", components[placeholder])
                else:
                    prompt_text = prompt_text.replace(f"{{{placeholder}}}", "")
            
            # Replace metadata placeholders
            for placeholder in template.metadata_placeholders:
                if placeholder in components:
                    prompt_text = prompt_text.replace(f"{{{placeholder}}}", components[placeholder])
                else:
                    prompt_text = prompt_text.replace(f"{{{placeholder}}}", "")
            
            # Apply style adaptations
            if self.config["enable_style_adaptation"]:
                prompt_text = await self._adapt_prompt_style(prompt_text, request)
            
            # Apply complexity adaptations
            if self.config["enable_adaptive_complexity"]:
                prompt_text = await self._adapt_prompt_complexity(prompt_text, request)
            
            # Final optimization
            if self.config["enable_prompt_optimization"]:
                prompt_text = await self._optimize_prompt(prompt_text)
            
            return prompt_text
            
        except Exception as e:
            logger.error(f"Failed to generate prompt text: {e}")
            return f"Please help with: {request.user_query}"
    
    async def _adapt_prompt_style(self, prompt_text: str, request: PromptRequest) -> str:
        """Adapt prompt style based on user preferences"""
        
        if request.preferred_style == "casual":
            # Make more conversational
            prompt_text = prompt_text.replace("Please provide", "Can you help me understand")
            prompt_text = prompt_text.replace("Your response should be", "I'd like you to")
        
        elif request.preferred_style == "academic":
            # Make more formal and academic
            prompt_text = prompt_text.replace("explain", "elucidate")
            prompt_text = prompt_text.replace("help", "assist")
        
        elif request.preferred_style == "technical":
            # Add technical focus
            if "technical" not in prompt_text.lower():
                prompt_text += "\n\nFocus on technical details and implementation specifics."
        
        return prompt_text
    
    async def _adapt_prompt_complexity(self, prompt_text: str, request: PromptRequest) -> str:
        """Adapt prompt complexity based on user expertise level"""
        
        if request.user_expertise_level == "beginner":
            # Simplify language and add explanatory notes
            additions = "\n\nPlease explain any technical terms and provide context for complex concepts."
            if additions not in prompt_text:
                prompt_text += additions
        
        elif request.user_expertise_level == "expert":
            # Increase depth and remove basic explanations
            additions = "\n\nAssuming expert-level knowledge, provide advanced insights and technical depth."
            if additions not in prompt_text:
                prompt_text += additions
        
        return prompt_text
    
    async def _optimize_prompt(self, prompt_text: str) -> str:
        """Apply final optimizations to prompt"""
        
        # Remove excessive whitespace
        prompt_text = re.sub(r'\n\s*\n\s*\n', '\n\n', prompt_text)
        prompt_text = re.sub(r' +', ' ', prompt_text)
        
        # Ensure proper formatting
        prompt_text = prompt_text.strip()
        
        # Ensure reasonable length
        if len(prompt_text) > self.config["max_prompt_length"]:
            # Truncate while preserving structure
            sections = prompt_text.split('\n\n')
            truncated_sections = []
            current_length = 0
            
            for section in sections:
                if current_length + len(section) <= self.config["max_prompt_length"]:
                    truncated_sections.append(section)
                    current_length += len(section)
                else:
                    break
            
            prompt_text = '\n\n'.join(truncated_sections)
            if len(prompt_text) < self.config["max_prompt_length"] - 100:
                prompt_text += "\n\n[Note: Content was truncated to fit length limits]"
        
        return prompt_text
    
    async def _calculate_quality_metrics(
        self, 
        prompt_text: str, 
        hierarchical_context: Dict[str, Any], 
        components: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate quality metrics for the generated prompt"""
        
        try:
            # Context coverage score
            context_content = hierarchical_context.get("context", "")
            if context_content:
                context_coverage = min(1.0, len(context_content) / 2000)  # Normalize by target length
            else:
                context_coverage = 0.0
            
            # Entity integration score
            entities = components.get("entities", [])
            entity_integration = min(1.0, len(entities) / 10) if entities else 0.0
            
            # Relationship integration score
            relationships = components.get("relationships", [])
            relationship_integration = min(1.0, len(relationships) / 5) if relationships else 0.0
            
            # Overall quality score (weighted average)
            weights = self.config
            overall_quality = (
                context_coverage * weights["context_weight"] +
                entity_integration * weights["entity_weight"] +
                relationship_integration * weights["relationship_weight"]
            )
            
            return {
                "context_coverage": context_coverage,
                "entity_integration": entity_integration,
                "relationship_integration": relationship_integration,
                "overall_quality": overall_quality
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate quality metrics: {e}")
            return {
                "context_coverage": 0.0,
                "entity_integration": 0.0,
                "relationship_integration": 0.0,
                "overall_quality": 0.0
            }
    
    def _estimate_response_length(self, prompt_text: str, complexity: PromptComplexity) -> int:
        """Estimate expected response length based on prompt and complexity"""
        
        base_length = len(prompt_text) * 0.8  # Assume response is roughly 80% of prompt length
        
        # Adjust by complexity
        complexity_multipliers = {
            PromptComplexity.SIMPLE: 0.5,
            PromptComplexity.MODERATE: 1.0,
            PromptComplexity.COMPLEX: 1.5,
            PromptComplexity.EXPERT: 2.0
        }
        
        multiplier = complexity_multipliers.get(complexity, 1.0)
        
        return int(base_length * multiplier)
    
    async def get_prompt_analytics(self) -> Dict[str, Any]:
        """Get analytics about prompt generation"""
        
        if not self.prompt_history:
            return {"total_prompts": 0}
        
        # Calculate statistics
        total_prompts = len(self.prompt_history)
        avg_quality = sum(p.overall_quality_score for p in self.prompt_history) / total_prompts
        
        # Group by type and complexity
        type_counts = defaultdict(int)
        complexity_counts = defaultdict(int)
        
        for prompt in self.prompt_history:
            type_counts[prompt.prompt_type.value] += 1
            complexity_counts[prompt.complexity.value] += 1
        
        return {
            "total_prompts": total_prompts,
            "average_quality_score": avg_quality,
            "prompt_types": dict(type_counts),
            "complexity_distribution": dict(complexity_counts),
            "templates_available": len(self.prompt_templates),
            "recent_prompts": len([p for p in self.prompt_history if (datetime.now() - p.generation_timestamp).days < 1])
        }

# Global enhanced prompt generator
_enhanced_prompt_generator = None

async def get_enhanced_prompt_generator() -> EnhancedPromptGenerator:
    """Get global enhanced prompt generator instance"""
    global _enhanced_prompt_generator
    
    if _enhanced_prompt_generator is None:
        _enhanced_prompt_generator = EnhancedPromptGenerator()
        await _enhanced_prompt_generator.initialize()
    
    return _enhanced_prompt_generator

if __name__ == "__main__":
    # Test enhanced prompt generator
    async def test_enhanced_prompt_generator():
        
        generator = EnhancedPromptGenerator()
        await generator.initialize()
        print("✅ Enhanced prompt generator created")
        
        # Test prompt request
        request = PromptRequest(
            user_query="How does machine learning work?",
            prompt_type=PromptType.EXPLANATION,
            complexity=PromptComplexity.MODERATE,
            integration_strategy=ContextIntegrationStrategy.HIERARCHICAL,
            include_entities=True,
            include_relationships=True
        )
        
        # Test template selection
        template = generator._select_template(request)
        print(f"✅ Template selected: {template.template_id}")
        
        # Test prompt generation (would need actual context)
        # For testing, we'll just verify the structure
        print(f"✅ Template placeholders: {len(template.context_placeholders)} context, {len(template.entity_placeholders)} entity")
        
        # Test analytics
        analytics = await generator.get_prompt_analytics()
        print(f"✅ Analytics: {analytics['templates_available']} templates available")
        
        return True
    
    import asyncio
    success = asyncio.run(test_enhanced_prompt_generator())
    if success:
        print("\n🎉 Enhanced prompt generator test completed successfully!")
    else:
        print("\n❌ Enhanced prompt generator test failed")