"""
Phase 3: Hierarchical Context Integration

Advanced context management system that leverages document hierarchies,
relationships, and semantic structures for enhanced AI interactions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of context information"""
    DOCUMENT_SECTION = "document_section"
    DOCUMENT_CHUNK = "document_chunk"
    DOCUMENT_TABLE = "document_table"
    DOCUMENT_ENTITY = "document_entity"
    CONVERSATION_THREAD = "conversation_thread"
    USER_INTERACTION = "user_interaction"

class HierarchyLevel(Enum):
    """Hierarchy levels in document structure"""
    DOCUMENT = "document"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    PHRASE = "phrase"

class RelationshipType(Enum):
    """Types of relationships between context elements"""
    PARENT_CHILD = "parent_child"
    SIBLING = "sibling"
    REFERENCE = "reference"
    DEPENDENCY = "dependency"
    SEQUENCE = "sequence"
    TOPIC_RELATION = "topic_relation"
    ENTITY_MENTION = "entity_mention"

@dataclass
class ContextElement:
    """Individual context element with hierarchical information"""
    id: str
    content: str
    context_type: ContextType
    hierarchy_level: HierarchyLevel
    
    # Hierarchical structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    sibling_ids: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0
    quality_score: float = 0.0
    
    # Relationships
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    
    # Content analysis
    entities: List[Dict[str, Any]] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Position information
    document_path: List[str] = field(default_factory=list)  # Path from root to this element
    sequence_position: Optional[int] = None
    depth_level: int = 0

@dataclass
class ContextHierarchy:
    """Hierarchical context structure"""
    root_elements: List[ContextElement] = field(default_factory=list)
    all_elements: Dict[str, ContextElement] = field(default_factory=dict)
    hierarchy_map: Dict[str, List[str]] = field(default_factory=dict)  # parent_id -> [child_ids]
    relationship_graph: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    
    # Context statistics
    total_elements: int = 0
    max_depth: int = 0
    coverage_stats: Dict[str, int] = field(default_factory=dict)

@dataclass
class ContextRequest:
    """Request for hierarchical context"""
    query: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Context preferences
    max_context_length: int = 4000
    preferred_context_types: List[ContextType] = field(default_factory=list)
    hierarchy_strategy: str = "adaptive"  # "breadth_first", "depth_first", "adaptive"
    include_relationships: bool = True
    include_entities: bool = True
    
    # Quality filters
    min_relevance_score: float = 0.3
    min_quality_score: float = 0.5
    max_elements: int = 20

class HierarchicalContextManager:
    """
    Advanced context management with hierarchical understanding
    """
    
    def __init__(self):
        self.context_hierarchy = ContextHierarchy()
        self.context_cache: Dict[str, Any] = {}
        self.relationship_analyzer = RelationshipAnalyzer()
        self.hierarchy_builder = HierarchyBuilder()
        self.context_optimizer = ContextOptimizer()
        
        # Configuration
        self.config = {
            "max_cache_size": 1000,
            "cache_ttl_seconds": 3600,
            "enable_relationship_analysis": True,
            "enable_hierarchy_optimization": True,
            "context_expansion_strategy": "intelligent",
            "quality_weighting": {
                "relevance": 0.4,
                "quality": 0.3,
                "hierarchy_position": 0.2,
                "recency": 0.1
            }
        }
        
        logger.info("Hierarchical Context Manager initialized")
    
    async def build_context_hierarchy(
        self, 
        search_results: List[Dict[str, Any]], 
        query: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContextHierarchy:
        """Build hierarchical context structure from search results"""
        
        try:
            # Create context elements from search results
            context_elements = []
            for result in search_results:
                element = await self._create_context_element(result, query)
                if element:
                    context_elements.append(element)
            
            # Build hierarchy structure
            hierarchy = await self.hierarchy_builder.build_hierarchy(context_elements)
            
            # Analyze relationships
            if self.config["enable_relationship_analysis"]:
                hierarchy = await self.relationship_analyzer.analyze_relationships(hierarchy)
            
            # Optimize context structure
            if self.config["enable_hierarchy_optimization"]:
                hierarchy = await self.context_optimizer.optimize_hierarchy(hierarchy, query)
            
            # Update statistics
            hierarchy.total_elements = len(hierarchy.all_elements)
            hierarchy.max_depth = self._calculate_max_depth(hierarchy)
            hierarchy.coverage_stats = self._calculate_coverage_stats(hierarchy)
            
            logger.info(f"Built context hierarchy: {hierarchy.total_elements} elements, max depth: {hierarchy.max_depth}")
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Failed to build context hierarchy: {e}")
            return ContextHierarchy()
    
    async def get_hierarchical_context(
        self, 
        request: ContextRequest
    ) -> Dict[str, Any]:
        """Get optimized hierarchical context for a request"""
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.context_cache:
                cached_result = self.context_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    logger.debug(f"Returning cached context for: {request.query[:50]}...")
                    return cached_result["context"]
            
            # Get initial search results (this would integrate with existing search)
            search_results = await self._get_search_results(request)
            
            # Build context hierarchy
            hierarchy = await self.build_context_hierarchy(search_results, request.query)
            
            # Select optimal context elements
            selected_context = await self._select_optimal_context(hierarchy, request)
            
            # Format context for AI consumption
            formatted_context = await self._format_hierarchical_context(selected_context, request)
            
            # Cache the result
            self._cache_context(cache_key, formatted_context)
            
            logger.info(f"Generated hierarchical context: {len(selected_context)} elements")
            
            return formatted_context
            
        except Exception as e:
            logger.error(f"Failed to get hierarchical context: {e}")
            return {"context": "", "metadata": {}, "hierarchy": []}
    
    async def _create_context_element(
        self, 
        search_result: Dict[str, Any], 
        query: str
    ) -> Optional[ContextElement]:
        """Create context element from search result"""
        
        try:
            # Extract basic information
            content = search_result.get("document", "")
            metadata = search_result.get("metadata", {})
            
            # Determine context type
            context_type = self._determine_context_type(metadata)
            
            # Determine hierarchy level
            hierarchy_level = self._determine_hierarchy_level(metadata, content)
            
            # Calculate relevance and quality scores
            relevance_score = self._calculate_relevance_score(content, query)
            quality_score = self._calculate_quality_score(content, metadata)
            
            # Extract entities and topics
            entities = await self._extract_entities(content)
            topics = await self._extract_topics(content)
            keywords = await self._extract_keywords(content, query)
            
            # Create element
            element = ContextElement(
                id=search_result.get("id", f"ctx_{hash(content)}"),
                content=content,
                context_type=context_type,
                hierarchy_level=hierarchy_level,
                metadata=metadata,
                relevance_score=relevance_score,
                quality_score=quality_score,
                entities=entities,
                topics=topics,
                keywords=keywords,
                document_path=self._extract_document_path(metadata),
                depth_level=self._calculate_depth_level(metadata)
            )
            
            return element
            
        except Exception as e:
            logger.error(f"Failed to create context element: {e}")
            return None
    
    def _determine_context_type(self, metadata: Dict[str, Any]) -> ContextType:
        """Determine context type from metadata"""
        
        # Check collection name or type field
        collection = metadata.get("collection", "")
        doc_type = metadata.get("type", "")
        
        if "section" in collection.lower() or "section" in doc_type.lower():
            return ContextType.DOCUMENT_SECTION
        elif "chunk" in collection.lower() or "chunk" in doc_type.lower():
            return ContextType.DOCUMENT_CHUNK
        elif "table" in collection.lower() or "table" in doc_type.lower():
            return ContextType.DOCUMENT_TABLE
        elif "entity" in collection.lower() or "entity" in doc_type.lower():
            return ContextType.DOCUMENT_ENTITY
        elif "conversation" in collection.lower():
            return ContextType.CONVERSATION_THREAD
        else:
            return ContextType.DOCUMENT_CHUNK  # Default
    
    def _determine_hierarchy_level(self, metadata: Dict[str, Any], content: str) -> HierarchyLevel:
        """Determine hierarchy level from metadata and content"""
        
        # Check explicit hierarchy level
        if "hierarchy_level" in metadata:
            level_map = {
                0: HierarchyLevel.DOCUMENT,
                1: HierarchyLevel.CHAPTER,
                2: HierarchyLevel.SECTION,
                3: HierarchyLevel.SUBSECTION,
                4: HierarchyLevel.PARAGRAPH,
                5: HierarchyLevel.SENTENCE
            }
            return level_map.get(metadata["hierarchy_level"], HierarchyLevel.PARAGRAPH)
        
        # Infer from content length and structure
        content_length = len(content)
        
        if content_length > 5000:
            return HierarchyLevel.CHAPTER
        elif content_length > 2000:
            return HierarchyLevel.SECTION
        elif content_length > 500:
            return HierarchyLevel.SUBSECTION
        elif content_length > 100:
            return HierarchyLevel.PARAGRAPH
        else:
            return HierarchyLevel.SENTENCE
    
    def _calculate_relevance_score(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query"""
        
        try:
            # Simple keyword matching (could be enhanced with embeddings)
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            # Calculate overlap
            overlap = len(query_words.intersection(content_words))
            max_possible = len(query_words)
            
            if max_possible == 0:
                return 0.0
            
            # Basic relevance score
            base_score = overlap / max_possible
            
            # Boost for exact phrase matches
            if query.lower() in content.lower():
                base_score += 0.3
            
            # Boost for title/header matches
            if any(word in content[:100].lower() for word in query_words):
                base_score += 0.2
            
            return min(1.0, base_score)
            
        except Exception:
            return 0.5  # Default score
    
    def _calculate_quality_score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate quality score for content"""
        
        try:
            score = 0.0
            
            # Content length score (not too short, not too long)
            length = len(content)
            if 100 <= length <= 2000:
                score += 0.3
            elif 50 <= length <= 5000:
                score += 0.2
            
            # Structure quality
            if re.search(r'^#+\s', content, re.MULTILINE):  # Has headers
                score += 0.2
            if re.search(r'\n\s*\n', content):  # Has paragraphs
                score += 0.1
            
            # Metadata quality
            if metadata.get("coherence_score", 0) > 0.7:
                score += 0.2
            if metadata.get("confidence", 0) > 0.8:
                score += 0.2
            
            return min(1.0, score)
            
        except Exception:
            return 0.5  # Default score
    
    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from content"""
        
        try:
            # Simple entity extraction (could be enhanced with NLP models)
            entities = []
            
            # Extract capitalized words as potential entities
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            
            for word in set(words):
                if len(word) > 2:  # Filter short words
                    entities.append({
                        "text": word,
                        "type": "UNKNOWN",
                        "confidence": 0.7
                    })
            
            return entities[:10]  # Limit to top 10
            
        except Exception:
            return []
    
    async def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from content"""
        
        try:
            # Simple topic extraction using keyword patterns
            topics = []
            
            # Look for topic indicators
            topic_patterns = [
                r'\b(?:about|regarding|concerning|on the topic of)\s+([^.!?]+)',
                r'\b(?:focuses on|deals with|covers)\s+([^.!?]+)',
                r'^#+\s*(.+)$'  # Headers as topics
            ]
            
            for pattern in topic_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                topics.extend([match.strip() for match in matches])
            
            return list(set(topics))[:5]  # Limit and deduplicate
            
        except Exception:
            return []
    
    async def _extract_keywords(self, content: str, query: str) -> List[str]:
        """Extract relevant keywords from content"""
        
        try:
            # Extract words that appear multiple times
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            word_freq = defaultdict(int)
            
            for word in words:
                word_freq[word] += 1
            
            # Filter and sort by frequency
            keywords = [word for word, freq in word_freq.items() if freq > 1]
            keywords.sort(key=lambda w: word_freq[w], reverse=True)
            
            # Boost query-related keywords
            query_words = set(query.lower().split())
            boosted_keywords = []
            
            for word in keywords:
                if word in query_words:
                    boosted_keywords.insert(0, word)
                else:
                    boosted_keywords.append(word)
            
            return boosted_keywords[:10]  # Top 10 keywords
            
        except Exception:
            return []
    
    def _extract_document_path(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract document hierarchy path"""
        
        path = []
        
        # Try to build path from metadata
        if "document_id" in metadata:
            path.append(metadata["document_id"])
        
        if "section_id" in metadata:
            path.append(metadata["section_id"])
        
        if "chunk_id" in metadata:
            path.append(metadata["chunk_id"])
        
        return path
    
    def _calculate_depth_level(self, metadata: Dict[str, Any]) -> int:
        """Calculate depth level in hierarchy"""
        
        # Use explicit depth if available
        if "depth" in metadata:
            return metadata["depth"]
        
        # Infer from path length
        path = self._extract_document_path(metadata)
        return len(path)
    
    async def _get_search_results(self, request: ContextRequest) -> List[Dict[str, Any]]:
        """Get search results for context request (placeholder - would integrate with search system)"""
        
        # This would integrate with the existing search system
        # For now, return empty list
        return []
    
    async def _select_optimal_context(
        self, 
        hierarchy: ContextHierarchy, 
        request: ContextRequest
    ) -> List[ContextElement]:
        """Select optimal context elements based on request parameters"""
        
        try:
            # Filter elements by quality thresholds
            candidates = [
                element for element in hierarchy.all_elements.values()
                if element.relevance_score >= request.min_relevance_score
                and element.quality_score >= request.min_quality_score
            ]
            
            # Filter by preferred context types
            if request.preferred_context_types:
                candidates = [
                    element for element in candidates
                    if element.context_type in request.preferred_context_types
                ]
            
            # Sort by composite score
            candidates.sort(key=lambda x: self._calculate_composite_score(x), reverse=True)
            
            # Apply hierarchy strategy
            selected = await self._apply_hierarchy_strategy(
                candidates, request.hierarchy_strategy, request.max_elements
            )
            
            # Ensure context length limit
            selected = self._enforce_context_length_limit(selected, request.max_context_length)
            
            logger.debug(f"Selected {len(selected)} context elements from {len(candidates)} candidates")
            
            return selected
            
        except Exception as e:
            logger.error(f"Failed to select optimal context: {e}")
            return []
    
    def _calculate_composite_score(self, element: ContextElement) -> float:
        """Calculate composite score for context element"""
        
        weights = self.config["quality_weighting"]
        
        # Base scores
        relevance_score = element.relevance_score * weights["relevance"]
        quality_score = element.quality_score * weights["quality"]
        
        # Hierarchy position score (higher levels get higher scores)
        hierarchy_scores = {
            HierarchyLevel.DOCUMENT: 1.0,
            HierarchyLevel.CHAPTER: 0.9,
            HierarchyLevel.SECTION: 0.8,
            HierarchyLevel.SUBSECTION: 0.7,
            HierarchyLevel.PARAGRAPH: 0.6,
            HierarchyLevel.SENTENCE: 0.5,
            HierarchyLevel.PHRASE: 0.4
        }
        hierarchy_score = hierarchy_scores.get(element.hierarchy_level, 0.5) * weights["hierarchy_position"]
        
        # Recency score (newer content gets slight boost)
        time_diff = (datetime.now() - element.timestamp).days
        recency_score = max(0.0, 1.0 - (time_diff / 365)) * weights["recency"]
        
        return relevance_score + quality_score + hierarchy_score + recency_score
    
    async def _apply_hierarchy_strategy(
        self, 
        candidates: List[ContextElement], 
        strategy: str, 
        max_elements: int
    ) -> List[ContextElement]:
        """Apply hierarchy selection strategy"""
        
        if strategy == "breadth_first":
            # Select elements from different hierarchy levels
            selected = []
            level_counts = defaultdict(int)
            max_per_level = max(1, max_elements // 6)  # Distribute across 6 levels
            
            for element in candidates:
                if len(selected) >= max_elements:
                    break
                if level_counts[element.hierarchy_level] < max_per_level:
                    selected.append(element)
                    level_counts[element.hierarchy_level] += 1
            
            return selected
        
        elif strategy == "depth_first":
            # Prefer deeper, more specific elements
            candidates.sort(key=lambda x: x.depth_level, reverse=True)
            return candidates[:max_elements]
        
        else:  # adaptive
            # Intelligent selection based on content diversity
            selected = []
            used_topics = set()
            used_documents = set()
            
            for element in candidates:
                if len(selected) >= max_elements:
                    break
                
                # Prefer elements that add topic diversity
                element_topics = set(element.topics)
                if not element_topics.intersection(used_topics) or len(selected) < max_elements // 2:
                    # Prefer elements from different documents
                    doc_path = element.document_path[0] if element.document_path else "unknown"
                    if doc_path not in used_documents or len(selected) < max_elements // 3:
                        selected.append(element)
                        used_topics.update(element_topics)
                        used_documents.add(doc_path)
            
            return selected
    
    def _enforce_context_length_limit(
        self, 
        elements: List[ContextElement], 
        max_length: int
    ) -> List[ContextElement]:
        """Enforce context length limit while preserving hierarchy"""
        
        if not elements:
            return elements
        
        # Calculate current total length
        total_length = sum(len(element.content) for element in elements)
        
        if total_length <= max_length:
            return elements
        
        # Remove elements starting from lowest scoring ones
        sorted_elements = sorted(elements, key=self._calculate_composite_score, reverse=True)
        selected = []
        current_length = 0
        
        for element in sorted_elements:
            if current_length + len(element.content) <= max_length:
                selected.append(element)
                current_length += len(element.content)
            else:
                # Try to fit a truncated version
                remaining_space = max_length - current_length
                if remaining_space > 100:  # Only if significant space remains
                    truncated_content = element.content[:remaining_space-3] + "..."
                    element.content = truncated_content
                    selected.append(element)
                break
        
        return selected
    
    async def _format_hierarchical_context(
        self, 
        elements: List[ContextElement], 
        request: ContextRequest
    ) -> Dict[str, Any]:
        """Format context elements for AI consumption"""
        
        try:
            # Group elements by hierarchy level
            hierarchy_groups = defaultdict(list)
            for element in elements:
                hierarchy_groups[element.hierarchy_level].append(element)
            
            # Build structured context
            context_parts = []
            metadata = {
                "total_elements": len(elements),
                "hierarchy_levels": ",".join(str(k) for k in hierarchy_groups.keys()),
                "context_types": ",".join(set(element.context_type for element in elements)),
                "total_length": sum(len(element.content) for element in elements)
            }
            
            # Add hierarchy information if requested
            if request.include_relationships:
                rel_summary = self._extract_relationship_summary(elements)
                metadata["parent_child_pairs"] = rel_summary["parent_child_pairs"]
                metadata["sibling_groups"] = rel_summary["sibling_groups"]
                metadata["cross_references"] = rel_summary["cross_references"]
            
            if request.include_entities:
                entity_summary = self._extract_entity_summary(elements)
                metadata["total_entities"] = entity_summary["total_entities"]
                metadata["unique_entities"] = entity_summary["unique_entities"]
                # Convert entity_types dict to string
                metadata["entity_types"] = ",".join(f"{k}:{v}" for k, v in entity_summary["entity_types"].items())
            
            # Format context by hierarchy level
            level_order = [
                HierarchyLevel.DOCUMENT,
                HierarchyLevel.CHAPTER,
                HierarchyLevel.SECTION,
                HierarchyLevel.SUBSECTION,
                HierarchyLevel.PARAGRAPH,
                HierarchyLevel.SENTENCE
            ]
            
            for level in level_order:
                if level in hierarchy_groups:
                    level_elements = hierarchy_groups[level]
                    level_elements.sort(key=self._calculate_composite_score, reverse=True)
                    
                    context_parts.append(f"\n## {level.value.title()} Context:")
                    for i, element in enumerate(level_elements, 1):
                        context_parts.append(f"\n{i}. {element.content}")
                        
                        # Add metadata if available
                        if element.topics:
                            context_parts.append(f"   Topics: {', '.join(element.topics[:3])}")
                        if element.keywords:
                            context_parts.append(f"   Keywords: {', '.join(element.keywords[:3])}")
            
            # Combine all context
            full_context = "\n".join(context_parts)
            
            return {
                "context": full_context,
                "metadata": metadata,
                "hierarchy": [
                    {
                        "id": element.id,
                        "level": element.hierarchy_level.value,
                        "type": element.context_type.value,
                        "score": self._calculate_composite_score(element),
                        "length": len(element.content)
                    }
                    for element in elements
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to format hierarchical context: {e}")
            return {"context": "", "metadata": {}, "hierarchy": []}
    
    def _extract_relationship_summary(self, elements: List[ContextElement]) -> Dict[str, Any]:
        """Extract relationship summary from elements"""
        
        relationships = {
            "parent_child_pairs": 0,
            "sibling_groups": 0,
            "cross_references": 0
        }
        
        # Count relationships
        for element in elements:
            if element.parent_id:
                relationships["parent_child_pairs"] += 1
            if len(element.sibling_ids) > 0:
                relationships["sibling_groups"] += 1
            if element.relationships:
                relationships["cross_references"] += len(element.relationships)
        
        return relationships
    
    def _extract_entity_summary(self, elements: List[ContextElement]) -> Dict[str, Any]:
        """Extract entity summary from elements"""
        
        all_entities = []
        for element in elements:
            all_entities.extend(element.entities)
        
        # Group by entity type
        entity_types = defaultdict(int)
        for entity in all_entities:
            entity_types[entity.get("type", "UNKNOWN")] += 1
        
        return {
            "total_entities": len(all_entities),
            "unique_entities": len(set(entity["text"] for entity in all_entities)),
            "entity_types": dict(entity_types)
        }
    
    def _calculate_max_depth(self, hierarchy: ContextHierarchy) -> int:
        """Calculate maximum depth in hierarchy"""
        return max(element.depth_level for element in hierarchy.all_elements.values()) if hierarchy.all_elements else 0
    
    def _calculate_coverage_stats(self, hierarchy: ContextHierarchy) -> Dict[str, int]:
        """Calculate coverage statistics"""
        
        stats = defaultdict(int)
        
        for element in hierarchy.all_elements.values():
            stats[element.context_type.value] += 1
            stats[element.hierarchy_level.value] += 1
        
        return dict(stats)
    
    def _generate_cache_key(self, request: ContextRequest) -> str:
        """Generate cache key for context request"""
        
        key_parts = [
            request.query,
            str(request.max_context_length),
            str(request.hierarchy_strategy),
            str(sorted([ct.value for ct in request.preferred_context_types]))
        ]
        
        return hash(tuple(key_parts))
    
    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached result is still valid"""
        
        cache_time = cached_result.get("timestamp", datetime.now())
        age_seconds = (datetime.now() - cache_time).total_seconds()
        
        return age_seconds < self.config["cache_ttl_seconds"]
    
    def _cache_context(self, cache_key: str, context: Dict[str, Any]):
        """Cache context result"""
        
        # Implement LRU cache behavior
        if len(self.context_cache) >= self.config["max_cache_size"]:
            # Remove oldest entry
            oldest_key = min(self.context_cache.keys(), 
                           key=lambda k: self.context_cache[k]["timestamp"])
            del self.context_cache[oldest_key]
        
        self.context_cache[cache_key] = {
            "context": context,
            "timestamp": datetime.now()
        }

class RelationshipAnalyzer:
    """Analyzes relationships between context elements"""
    
    async def analyze_relationships(self, hierarchy: ContextHierarchy) -> ContextHierarchy:
        """Analyze and establish relationships between context elements"""
        
        try:
            # Analyze parent-child relationships
            await self._analyze_hierarchical_relationships(hierarchy)
            
            # Analyze content-based relationships
            await self._analyze_content_relationships(hierarchy)
            
            # Analyze sequence relationships
            await self._analyze_sequence_relationships(hierarchy)
            
            logger.debug(f"Analyzed relationships for {len(hierarchy.all_elements)} elements")
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Failed to analyze relationships: {e}")
            return hierarchy
    
    async def _analyze_hierarchical_relationships(self, hierarchy: ContextHierarchy):
        """Analyze hierarchical parent-child relationships"""
        
        # Group elements by document path
        path_groups = defaultdict(list)
        for element in hierarchy.all_elements.values():
            if element.document_path:
                path_key = "/".join(element.document_path[:-1]) if len(element.document_path) > 1 else "root"
                path_groups[path_key].append(element)
        
        # Establish parent-child relationships
        for path, elements in path_groups.items():
            # Sort by depth level
            elements.sort(key=lambda x: x.depth_level)
            
            for i, element in enumerate(elements):
                # Find parent (element with shorter path)
                for potential_parent in elements[:i]:
                    if (len(potential_parent.document_path) < len(element.document_path) and
                        all(a == b for a, b in zip(potential_parent.document_path, element.document_path))):
                        element.parent_id = potential_parent.id
                        potential_parent.children_ids.append(element.id)
                        break
    
    async def _analyze_content_relationships(self, hierarchy: ContextHierarchy):
        """Analyze content-based relationships between elements"""
        
        elements = list(hierarchy.all_elements.values())
        
        for i, element1 in enumerate(elements):
            for element2 in elements[i+1:]:
                # Check for entity overlap
                entities1 = set(entity["text"] for entity in element1.entities)
                entities2 = set(entity["text"] for entity in element2.entities)
                
                entity_overlap = len(entities1.intersection(entities2))
                if entity_overlap > 0:
                    if RelationshipType.ENTITY_MENTION.value not in element1.relationships:
                        element1.relationships[RelationshipType.ENTITY_MENTION.value] = []
                    element1.relationships[RelationshipType.ENTITY_MENTION.value].append(element2.id)
                
                # Check for topic overlap
                topics1 = set(element1.topics)
                topics2 = set(element2.topics)
                
                topic_overlap = len(topics1.intersection(topics2))
                if topic_overlap > 0:
                    if RelationshipType.TOPIC_RELATION.value not in element1.relationships:
                        element1.relationships[RelationshipType.TOPIC_RELATION.value] = []
                    element1.relationships[RelationshipType.TOPIC_RELATION.value].append(element2.id)
    
    async def _analyze_sequence_relationships(self, hierarchy: ContextHierarchy):
        """Analyze sequence relationships between elements"""
        
        # Group by document and sort by position
        doc_groups = defaultdict(list)
        for element in hierarchy.all_elements.values():
            doc_id = element.document_path[0] if element.document_path else "unknown"
            doc_groups[doc_id].append(element)
        
        for doc_id, elements in doc_groups.items():
            # Sort by sequence position or depth level
            elements.sort(key=lambda x: (x.sequence_position or 0, x.depth_level))
            
            # Establish sequence relationships
            for i, element in enumerate(elements):
                element.sequence_position = i
                
                # Set sibling relationships
                if i > 0:
                    prev_element = elements[i-1]
                    if prev_element.hierarchy_level == element.hierarchy_level:
                        element.sibling_ids.append(prev_element.id)
                        prev_element.sibling_ids.append(element.id)

class HierarchyBuilder:
    """Builds hierarchical structure from context elements"""
    
    async def build_hierarchy(self, elements: List[ContextElement]) -> ContextHierarchy:
        """Build hierarchy structure from context elements"""
        
        hierarchy = ContextHierarchy()
        
        # Add all elements to the hierarchy
        for element in elements:
            hierarchy.all_elements[element.id] = element
        
        # Identify root elements (elements without parents)
        for element in elements:
            if not element.parent_id or element.parent_id not in hierarchy.all_elements:
                hierarchy.root_elements.append(element)
        
        # Build hierarchy map
        for element in elements:
            if element.parent_id and element.parent_id in hierarchy.all_elements:
                if element.parent_id not in hierarchy.hierarchy_map:
                    hierarchy.hierarchy_map[element.parent_id] = []
                hierarchy.hierarchy_map[element.parent_id].append(element.id)
        
        logger.debug(f"Built hierarchy: {len(hierarchy.root_elements)} root elements")
        
        return hierarchy

class ContextOptimizer:
    """Optimizes context structure for better AI consumption"""
    
    async def optimize_hierarchy(self, hierarchy: ContextHierarchy, query: str) -> ContextHierarchy:
        """Optimize hierarchy structure for AI consumption"""
        
        try:
            # Remove redundant elements
            hierarchy = await self._remove_redundant_elements(hierarchy)
            
            # Enhance element quality scores
            hierarchy = await self._enhance_quality_scores(hierarchy, query)
            
            # Optimize content for readability
            hierarchy = await self._optimize_content_readability(hierarchy)
            
            logger.debug("Optimized hierarchy structure")
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Failed to optimize hierarchy: {e}")
            return hierarchy
    
    async def _remove_redundant_elements(self, hierarchy: ContextHierarchy) -> ContextHierarchy:
        """Remove redundant or low-quality elements"""
        
        # Remove elements with very low scores
        min_threshold = 0.1
        elements_to_remove = []
        
        for element_id, element in hierarchy.all_elements.items():
            combined_score = (element.relevance_score + element.quality_score) / 2
            if combined_score < min_threshold:
                elements_to_remove.append(element_id)
        
        # Remove elements
        for element_id in elements_to_remove:
            del hierarchy.all_elements[element_id]
            hierarchy.root_elements = [e for e in hierarchy.root_elements if e.id != element_id]
        
        return hierarchy
    
    async def _enhance_quality_scores(self, hierarchy: ContextHierarchy, query: str) -> ContextHierarchy:
        """Enhance quality scores based on context"""
        
        query_words = set(query.lower().split())
        
        for element in hierarchy.all_elements.values():
            # Boost score for query-relevant content
            element_words = set(element.content.lower().split())
            overlap_ratio = len(query_words.intersection(element_words)) / len(query_words)
            
            if overlap_ratio > 0.5:
                element.quality_score = min(1.0, element.quality_score + 0.2)
            
            # Boost score for structured content
            if any(marker in element.content for marker in ['##', '**', '1.', '2.', '•']):
                element.quality_score = min(1.0, element.quality_score + 0.1)
        
        return hierarchy
    
    async def _optimize_content_readability(self, hierarchy: ContextHierarchy) -> ContextHierarchy:
        """Optimize content for better readability"""
        
        for element in hierarchy.all_elements.values():
            # Clean up content formatting
            content = element.content
            
            # Remove excessive whitespace
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            content = re.sub(r' +', ' ', content)
            
            # Ensure proper sentence endings
            content = re.sub(r'([.!?])([A-Z])', r'\1 \2', content)
            
            element.content = content.strip()
        
        return hierarchy

# Global hierarchical context manager
_hierarchical_context_manager = None

async def get_hierarchical_context_manager() -> HierarchicalContextManager:
    """Get global hierarchical context manager instance"""
    global _hierarchical_context_manager
    
    if _hierarchical_context_manager is None:
        _hierarchical_context_manager = HierarchicalContextManager()
    
    return _hierarchical_context_manager

if __name__ == "__main__":
    # Test hierarchical context system
    async def test_hierarchical_context():
        
        manager = HierarchicalContextManager()
        print("✅ Hierarchical context manager created")
        
        # Test context request
        request = ContextRequest(
            query="machine learning algorithms",
            max_context_length=2000,
            hierarchy_strategy="adaptive",
            include_relationships=True,
            include_entities=True
        )
        
        # Test context element creation
        sample_result = {
            "id": "test_1",
            "document": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "metadata": {
                "collection": "document_sections",
                "hierarchy_level": 2,
                "document_id": "ml_guide"
            }
        }
        
        element = await manager._create_context_element(sample_result, request.query)
        print(f"✅ Context element created: {element.hierarchy_level.value}")
        
        # Test hierarchy building
        elements = [element] if element else []
        hierarchy = await manager.build_context_hierarchy([sample_result], request.query)
        print(f"✅ Hierarchy built: {hierarchy.total_elements} elements")
        
        return True
    
    import asyncio
    success = asyncio.run(test_hierarchical_context())
    if success:
        print("\n🎉 Hierarchical context system test completed successfully!")
    else:
        print("\n❌ Hierarchical context system test failed")