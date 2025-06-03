"""
Production Knowledge Graph System

Implements entity extraction, relationship mapping, and knowledge graph
construction for enhanced search and AI integration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import json

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntity:
    """Represents an entity in the knowledge graph"""
    entity_id: str
    text: str
    entity_type: str
    
    # Source information
    document_ids: Set[str] = field(default_factory=set)
    section_ids: Set[str] = field(default_factory=set)
    chunk_ids: Set[str] = field(default_factory=set)
    
    # Entity properties
    frequency: int = 0
    confidence: float = 0.0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    # Attributes and metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    context_examples: List[str] = field(default_factory=list)


@dataclass 
class EntityRelationship:
    """Represents a relationship between entities"""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    
    # Relationship properties
    confidence: float = 0.0
    frequency: int = 0
    
    # Context and evidence
    evidence_contexts: List[str] = field(default_factory=list)
    document_ids: Set[str] = field(default_factory=set)
    
    # Metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConceptCluster:
    """Represents a cluster of related concepts"""
    cluster_id: str
    name: str
    entity_ids: Set[str] = field(default_factory=set)
    
    # Cluster properties
    coherence_score: float = 0.0
    centroid_entity_id: Optional[str] = None
    
    # Metadata
    keywords: List[str] = field(default_factory=list)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeGraphStats:
    """Statistics about the knowledge graph"""
    total_entities: int = 0
    total_relationships: int = 0
    total_clusters: int = 0
    
    entity_types: Dict[str, int] = field(default_factory=dict)
    relationship_types: Dict[str, int] = field(default_factory=dict)
    
    avg_relationships_per_entity: float = 0.0
    graph_density: float = 0.0
    
    most_connected_entities: List[Tuple[str, int]] = field(default_factory=list)
    largest_clusters: List[Tuple[str, int]] = field(default_factory=list)


class KnowledgeGraphSystem:
    """
    Production knowledge graph system for entity relationships and concept mapping
    """
    
    def __init__(self):
        # Core components
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.relationships: Dict[str, EntityRelationship] = {}
        self.clusters: Dict[str, ConceptCluster] = {}
        
        # Indexes for fast lookup
        self.entity_text_index: Dict[str, Set[str]] = defaultdict(set)  # text -> entity_ids
        self.entity_type_index: Dict[str, Set[str]] = defaultdict(set)  # type -> entity_ids
        self.document_entity_index: Dict[str, Set[str]] = defaultdict(set)  # doc_id -> entity_ids
        
        # NetworkX graph for analysis
        self.graph = nx.DiGraph()
        
        # Configuration
        self.config = {
            "min_entity_frequency": 2,
            "min_relationship_confidence": 0.5,
            "max_context_examples": 5,
            "clustering_threshold": 0.7,
            "max_cluster_size": 50,
            "relationship_co_occurrence_window": 3  # sentences
        }
        
        # Relationship type definitions
        self.relationship_types = {
            "RELATED_TO": {"weight": 1.0, "symmetric": True},
            "PART_OF": {"weight": 1.5, "symmetric": False},
            "SYNONYM": {"weight": 2.0, "symmetric": True},
            "OPPOSITE": {"weight": 1.0, "symmetric": True},
            "CAUSES": {"weight": 1.5, "symmetric": False},
            "PRECEDES": {"weight": 1.2, "symmetric": False},
            "LOCATED_IN": {"weight": 1.3, "symmetric": False},
            "WORKS_FOR": {"weight": 1.4, "symmetric": False}
        }
        
        logger.info("Knowledge Graph System initialized")
    
    async def add_document_entities(
        self,
        document_id: str,
        entities: List[Any],  # UnifiedDocumentEntity from processor
        sections: List[Any],  # UnifiedDocumentSection
        chunks: List[Any]     # UnifiedDocumentChunk
    ):
        """Add entities from a processed document to the knowledge graph"""
        logger.info(f"Adding entities from document {document_id}")
        
        # Build context lookup
        section_content = {s.section_id: s.content for s in sections}
        chunk_content = {c.chunk_id: c.content for c in chunks}
        
        # Process entities
        for entity in entities:
            # Get or create knowledge entity
            knowledge_entity = await self._get_or_create_entity(
                entity.text,
                entity.entity_type,
                entity.confidence
            )
            
            # Update entity with document info
            knowledge_entity.document_ids.add(document_id)
            knowledge_entity.section_ids.add(entity.section_id)
            if entity.chunk_id:
                knowledge_entity.chunk_ids.add(entity.chunk_id)
            
            # Update frequency and confidence
            knowledge_entity.frequency += 1
            knowledge_entity.confidence = max(knowledge_entity.confidence, entity.confidence)
            knowledge_entity.last_seen = datetime.now()
            
            # Add context example
            context = self._extract_entity_context(
                entity,
                section_content.get(entity.section_id, ""),
                chunk_content.get(entity.chunk_id, "") if entity.chunk_id else None
            )
            if context and len(knowledge_entity.context_examples) < self.config["max_context_examples"]:
                knowledge_entity.context_examples.append(context)
        
        # Extract relationships between entities
        await self._extract_relationships(document_id, entities, chunks)
        
        # Update graph
        self._update_graph()
        
        logger.info(f"Added {len(entities)} entities from document {document_id}")
    
    async def _get_or_create_entity(
        self,
        text: str,
        entity_type: str,
        confidence: float
    ) -> KnowledgeEntity:
        """Get existing entity or create new one"""
        # Normalize text for matching
        normalized_text = text.lower().strip()
        
        # Check if entity exists
        for entity_id in self.entity_text_index.get(normalized_text, set()):
            entity = self.entities[entity_id]
            if entity.entity_type == entity_type:
                return entity
        
        # Create new entity
        entity_id = f"ke_{len(self.entities)}_{hash(normalized_text)}"
        entity = KnowledgeEntity(
            entity_id=entity_id,
            text=text,
            entity_type=entity_type,
            confidence=confidence
        )
        
        # Add to storage and indexes
        self.entities[entity_id] = entity
        self.entity_text_index[normalized_text].add(entity_id)
        self.entity_type_index[entity_type].add(entity_id)
        
        return entity
    
    def _extract_entity_context(
        self,
        entity: Any,
        section_content: str,
        chunk_content: Optional[str]
    ) -> Optional[str]:
        """Extract context around entity mention"""
        # Use chunk content if available, otherwise section
        content = chunk_content if chunk_content else section_content
        
        if not content:
            return None
        
        # Find entity in content
        start = content.lower().find(entity.text.lower())
        if start == -1:
            return None
        
        # Extract surrounding context (50 chars before and after)
        context_start = max(0, start - 50)
        context_end = min(len(content), start + len(entity.text) + 50)
        
        context = content[context_start:context_end]
        
        # Clean up context
        if context_start > 0:
            context = "..." + context
        if context_end < len(content):
            context = context + "..."
        
        return context.strip()
    
    async def _extract_relationships(
        self,
        document_id: str,
        entities: List[Any],
        chunks: List[Any]
    ):
        """Extract relationships between entities"""
        # Group entities by chunk
        chunk_entities = defaultdict(list)
        for entity in entities:
            if entity.chunk_id:
                chunk_entities[entity.chunk_id].append(entity)
        
        # Find co-occurring entities within chunks
        for chunk_id, chunk_entity_list in chunk_entities.items():
            if len(chunk_entity_list) < 2:
                continue
            
            # Get chunk content
            chunk = next((c for c in chunks if c.chunk_id == chunk_id), None)
            if not chunk:
                continue
            
            # Check each pair of entities
            for i, entity1 in enumerate(chunk_entity_list):
                for entity2 in chunk_entity_list[i+1:]:
                    # Skip if same entity
                    if entity1.text.lower() == entity2.text.lower():
                        continue
                    
                    # Determine relationship type
                    rel_type, confidence = self._infer_relationship_type(
                        entity1, entity2, chunk.content
                    )
                    
                    if confidence >= self.config["min_relationship_confidence"]:
                        # Get knowledge entities
                        ke1 = await self._get_or_create_entity(
                            entity1.text, entity1.entity_type, entity1.confidence
                        )
                        ke2 = await self._get_or_create_entity(
                            entity2.text, entity2.entity_type, entity2.confidence
                        )
                        
                        # Create relationship
                        await self._create_or_update_relationship(
                            ke1.entity_id,
                            ke2.entity_id,
                            rel_type,
                            confidence,
                            chunk.content,
                            document_id
                        )
    
    def _infer_relationship_type(
        self,
        entity1: Any,
        entity2: Any,
        context: str
    ) -> Tuple[str, float]:
        """Infer relationship type between two entities"""
        # Simple heuristic-based relationship inference
        # In production, this would use more sophisticated NLP
        
        context_lower = context.lower()
        e1_text = entity1.text.lower()
        e2_text = entity2.text.lower()
        
        # Check for specific patterns
        patterns = [
            (r"part of|belongs to|component of", "PART_OF", 0.8),
            (r"located in|based in|found in", "LOCATED_IN", 0.8),
            (r"works for|employed by|employee of", "WORKS_FOR", 0.8),
            (r"causes|leads to|results in", "CAUSES", 0.7),
            (r"before|precedes|followed by", "PRECEDES", 0.7),
            (r"same as|also known as|aka", "SYNONYM", 0.9),
            (r"opposite of|contrary to|versus", "OPPOSITE", 0.8)
        ]
        
        for pattern, rel_type, confidence in patterns:
            # Check if pattern exists between entities
            import re
            if re.search(f"{e1_text}.*{pattern}.*{e2_text}", context_lower):
                return rel_type, confidence
            if re.search(f"{e2_text}.*{pattern}.*{e1_text}", context_lower):
                # Reverse relationship if needed
                if self.relationship_types[rel_type]["symmetric"]:
                    return rel_type, confidence
                # Handle non-symmetric relationships
                # This is simplified - would need proper inverse mapping
        
        # Default to general relationship
        return "RELATED_TO", 0.6
    
    async def _create_or_update_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        confidence: float,
        context: str,
        document_id: str
    ):
        """Create or update a relationship between entities"""
        # Create relationship ID
        rel_id = f"{source_id}_{rel_type}_{target_id}"
        
        if rel_id in self.relationships:
            # Update existing relationship
            rel = self.relationships[rel_id]
            rel.frequency += 1
            rel.confidence = max(rel.confidence, confidence)
            rel.document_ids.add(document_id)
            if context not in rel.evidence_contexts:
                rel.evidence_contexts.append(context[:200])  # Limit context length
        else:
            # Create new relationship
            rel = EntityRelationship(
                relationship_id=rel_id,
                source_entity_id=source_id,
                target_entity_id=target_id,
                relationship_type=rel_type,
                confidence=confidence,
                frequency=1,
                evidence_contexts=[context[:200]],
                document_ids={document_id}
            )
            self.relationships[rel_id] = rel
    
    def _update_graph(self):
        """Update NetworkX graph with current entities and relationships"""
        # Clear and rebuild graph
        self.graph.clear()
        
        # Add nodes
        for entity_id, entity in self.entities.items():
            self.graph.add_node(
                entity_id,
                text=entity.text,
                type=entity.entity_type,
                frequency=entity.frequency,
                confidence=entity.confidence
            )
        
        # Add edges
        for rel_id, rel in self.relationships.items():
            weight = self.relationship_types.get(
                rel.relationship_type, {"weight": 1.0}
            )["weight"]
            
            self.graph.add_edge(
                rel.source_entity_id,
                rel.target_entity_id,
                relationship_type=rel.relationship_type,
                confidence=rel.confidence,
                frequency=rel.frequency,
                weight=weight * rel.confidence
            )
    
    async def find_related_entities(
        self,
        entity_text: str,
        max_hops: int = 2,
        limit: int = 10
    ) -> List[Tuple[KnowledgeEntity, float]]:
        """Find entities related to the given entity"""
        # Normalize text
        normalized_text = entity_text.lower().strip()
        
        # Find matching entities
        matching_entity_ids = self.entity_text_index.get(normalized_text, set())
        if not matching_entity_ids:
            return []
        
        # Get all related entities within max_hops
        related_entities = set()
        relevance_scores = {}
        
        for entity_id in matching_entity_ids:
            if entity_id not in self.graph:
                continue
            
            # Use BFS to find entities within max_hops
            for node, distance in nx.single_source_shortest_path_length(
                self.graph, entity_id, cutoff=max_hops
            ).items():
                if node != entity_id:
                    related_entities.add(node)
                    # Calculate relevance score (inverse of distance)
                    score = 1.0 / (distance + 1)
                    relevance_scores[node] = max(
                        relevance_scores.get(node, 0), score
                    )
        
        # Get entities and sort by relevance
        results = []
        for entity_id in related_entities:
            if entity_id in self.entities:
                entity = self.entities[entity_id]
                score = relevance_scores[entity_id]
                results.append((entity, score))
        
        # Sort by score and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    async def find_concept_clusters(
        self,
        entity_texts: List[str]
    ) -> List[ConceptCluster]:
        """Find concept clusters related to given entities"""
        # Find matching entities
        entity_ids = set()
        for text in entity_texts:
            normalized = text.lower().strip()
            entity_ids.update(self.entity_text_index.get(normalized, set()))
        
        # Find clusters containing these entities
        relevant_clusters = []
        for cluster in self.clusters.values():
            overlap = len(entity_ids.intersection(cluster.entity_ids))
            if overlap > 0:
                # Calculate relevance score
                score = overlap / len(cluster.entity_ids)
                relevant_clusters.append((cluster, score))
        
        # Sort by relevance
        relevant_clusters.sort(key=lambda x: x[1], reverse=True)
        
        return [cluster for cluster, _ in relevant_clusters]
    
    async def build_concept_clusters(self):
        """Build concept clusters from entity relationships"""
        logger.info("Building concept clusters")
        
        # Use community detection on the graph
        if len(self.graph) == 0:
            return
        
        # Convert to undirected for community detection
        undirected_graph = self.graph.to_undirected()
        
        # Use Louvain community detection
        import community as community_louvain
        partition = community_louvain.best_partition(undirected_graph)
        
        # Create clusters from communities
        communities = defaultdict(set)
        for node, comm_id in partition.items():
            communities[comm_id].add(node)
        
        # Create concept clusters
        self.clusters.clear()
        
        for comm_id, entity_ids in communities.items():
            if len(entity_ids) < 2:  # Skip single-entity clusters
                continue
            
            if len(entity_ids) > self.config["max_cluster_size"]:
                # Split large clusters
                # This is simplified - in production would use hierarchical clustering
                continue
            
            # Create cluster
            cluster_id = f"cluster_{len(self.clusters)}"
            
            # Find cluster name (most frequent entity)
            entity_frequencies = [
                (self.entities[eid].frequency, self.entities[eid].text, eid)
                for eid in entity_ids if eid in self.entities
            ]
            entity_frequencies.sort(reverse=True)
            
            cluster_name = entity_frequencies[0][1] if entity_frequencies else f"Cluster {comm_id}"
            centroid_id = entity_frequencies[0][2] if entity_frequencies else None
            
            # Extract keywords
            keywords = []
            for _, text, _ in entity_frequencies[:5]:
                keywords.append(text)
            
            cluster = ConceptCluster(
                cluster_id=cluster_id,
                name=cluster_name,
                entity_ids=entity_ids,
                centroid_entity_id=centroid_id,
                keywords=keywords,
                coherence_score=self._calculate_cluster_coherence(entity_ids)
            )
            
            self.clusters[cluster_id] = cluster
        
        logger.info(f"Created {len(self.clusters)} concept clusters")
    
    def _calculate_cluster_coherence(self, entity_ids: Set[str]) -> float:
        """Calculate coherence score for a cluster"""
        if len(entity_ids) < 2:
            return 0.0
        
        # Calculate based on internal connectivity
        internal_edges = 0
        possible_edges = len(entity_ids) * (len(entity_ids) - 1) / 2
        
        for e1_id in entity_ids:
            for e2_id in entity_ids:
                if e1_id != e2_id and self.graph.has_edge(e1_id, e2_id):
                    internal_edges += 1
        
        # Coherence is ratio of actual to possible internal edges
        coherence = internal_edges / possible_edges if possible_edges > 0 else 0.0
        return coherence
    
    def extract_entities(self, text: str) -> List[KnowledgeEntity]:
        """Extract entities from text using simple regex/spacy approach"""
        entities = []
        
        try:
            # Simple entity extraction (placeholder implementation)
            # In a real implementation, this would use spacy or similar NLP library
            
            # Extract simple patterns for names (capitalized words)
            import re
            
            # Find capitalized words that might be names/entities
            words = text.split()
            for i, word in enumerate(words):
                # Clean word
                clean_word = re.sub(r'[^\w\s]', '', word)
                
                # Check if it looks like an entity (capitalized, not first word of sentence)
                if (clean_word and 
                    clean_word[0].isupper() and 
                    len(clean_word) > 2 and
                    clean_word.lower() not in ['the', 'and', 'but', 'for', 'are', 'this', 'that']):
                    
                    # Create entity
                    entity = KnowledgeEntity(
                        entity_id=f"extracted_{len(entities)}",
                        text=clean_word,
                        entity_type="GENERAL",
                        confidence=0.7
                    )
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def get_statistics(self) -> KnowledgeGraphStats:
        """Get current knowledge graph statistics"""
        stats = KnowledgeGraphStats()
        
        # Basic counts
        stats.total_entities = len(self.entities)
        stats.total_relationships = len(self.relationships)
        stats.total_clusters = len(self.clusters)
        
        # Entity type distribution
        for entity in self.entities.values():
            stats.entity_types[entity.entity_type] = \
                stats.entity_types.get(entity.entity_type, 0) + 1
        
        # Relationship type distribution
        for rel in self.relationships.values():
            stats.relationship_types[rel.relationship_type] = \
                stats.relationship_types.get(rel.relationship_type, 0) + 1
        
        # Graph metrics
        if len(self.graph) > 0:
            stats.avg_relationships_per_entity = \
                self.graph.number_of_edges() / self.graph.number_of_nodes()
            stats.graph_density = nx.density(self.graph)
            
            # Most connected entities
            degree_centrality = nx.degree_centrality(self.graph)
            sorted_nodes = sorted(
                degree_centrality.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            stats.most_connected_entities = [
                (self.entities[node_id].text, centrality)
                for node_id, centrality in sorted_nodes
                if node_id in self.entities
            ]
        
        # Largest clusters
        if self.clusters:
            sorted_clusters = sorted(
                self.clusters.values(),
                key=lambda c: len(c.entity_ids),
                reverse=True
            )[:5]
            
            stats.largest_clusters = [
                (cluster.name, len(cluster.entity_ids))
                for cluster in sorted_clusters
            ]
        
        return stats
    
    def export_graph(self, format: str = "json") -> str:
        """Export knowledge graph in specified format"""
        if format == "json":
            # Export as JSON
            graph_data = {
                "entities": [
                    {
                        "id": e.entity_id,
                        "text": e.text,
                        "type": e.entity_type,
                        "frequency": e.frequency,
                        "confidence": e.confidence
                    }
                    for e in self.entities.values()
                ],
                "relationships": [
                    {
                        "source": r.source_entity_id,
                        "target": r.target_entity_id,
                        "type": r.relationship_type,
                        "confidence": r.confidence,
                        "frequency": r.frequency
                    }
                    for r in self.relationships.values()
                ],
                "clusters": [
                    {
                        "id": c.cluster_id,
                        "name": c.name,
                        "entities": list(c.entity_ids),
                        "keywords": c.keywords
                    }
                    for c in self.clusters.values()
                ]
            }
            return json.dumps(graph_data, indent=2)
        
        elif format == "graphml":
            # Export as GraphML for visualization tools
            return nx.write_graphml(self.graph)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global knowledge graph instance
_knowledge_graph = None


def get_knowledge_graph() -> KnowledgeGraphSystem:
    """Get global knowledge graph instance"""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraphSystem()
    return _knowledge_graph


# Test function
if __name__ == "__main__":
    async def test_knowledge_graph():
        kg = KnowledgeGraphSystem()
        
        # Mock entities
        from unified_docling_processor import UnifiedDocumentEntity
        
        test_entities = [
            UnifiedDocumentEntity(
                entity_id="e1",
                text="John Smith",
                entity_type="PERSON",
                confidence=0.9,
                section_id="sec1",
                chunk_id="chunk1"
            ),
            UnifiedDocumentEntity(
                entity_id="e2",
                text="Acme Corporation",
                entity_type="ORG",
                confidence=0.85,
                section_id="sec1",
                chunk_id="chunk1"
            ),
            UnifiedDocumentEntity(
                entity_id="e3",
                text="Machine Learning",
                entity_type="CONCEPT",
                confidence=0.8,
                section_id="sec2",
                chunk_id="chunk2"
            )
        ]
        
        # Mock chunks
        from unified_docling_processor import UnifiedDocumentChunk
        test_chunks = [
            UnifiedDocumentChunk(
                chunk_id="chunk1",
                content="John Smith works for Acme Corporation as a senior engineer.",
                chunk_index=0,
                section_id="sec1"
            ),
            UnifiedDocumentChunk(
                chunk_id="chunk2",
                content="Machine Learning is a part of artificial intelligence.",
                chunk_index=1,
                section_id="sec2"
            )
        ]
        
        # Add to knowledge graph
        await kg.add_document_entities(
            "doc1",
            test_entities,
            [],  # sections
            test_chunks
        )
        
        # Get statistics
        stats = kg.get_statistics()
        print(f"Knowledge Graph Stats:")
        print(f"  Entities: {stats.total_entities}")
        print(f"  Relationships: {stats.total_relationships}")
        print(f"  Entity types: {stats.entity_types}")
        
        # Find related entities
        related = await kg.find_related_entities("John Smith", max_hops=2)
        print(f"\nEntities related to 'John Smith':")
        for entity, score in related:
            print(f"  - {entity.text} ({entity.entity_type}): {score:.2f}")
        
        # Build clusters
        await kg.build_concept_clusters()
        print(f"\nClusters created: {len(kg.clusters)}")
        
        return True
    
    # Run test
    import asyncio
    asyncio.run(test_knowledge_graph())