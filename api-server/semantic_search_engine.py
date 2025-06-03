"""
Enhanced Multi-Collection Search System - Production Implementation

Provides advanced search capabilities across multiple collections with knowledge graph
integration, semantic understanding, and intelligent result ranking.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import chromadb
from sentence_transformers import SentenceTransformer

from entity_relationship_graph import get_knowledge_graph

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SearchQuery:
    """Enhanced search query with analysis"""
    raw_query: str
    normalized_query: str
    search_type: str = "hybrid"  # keyword, semantic, hybrid
    
    # Query understanding
    intent: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    temporal_context: Optional[str] = None
    
    # Search parameters
    collections: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    offset: int = 0
    
    # Advanced options
    use_knowledge_graph: bool = True
    expand_query: bool = True
    include_context: bool = True
    min_relevance_score: float = 0.5


@dataclass
class SearchResult:
    """Enhanced search result with context"""
    result_id: str
    score: float
    
    # Content
    content: str
    content_type: str
    collection: str
    
    # Source information
    document_id: str
    document_title: Optional[str] = None
    section_title: Optional[str] = None
    chunk_index: Optional[int] = None
    
    # Context
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    full_context: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Relevance information
    keyword_matches: List[str] = field(default_factory=list)
    semantic_similarity: float = 0.0
    entity_matches: List[str] = field(default_factory=list)
    knowledge_graph_boost: float = 0.0
    
    # Highlighting
    highlighted_content: Optional[str] = None
    highlighted_context: Optional[str] = None


@dataclass
class SearchResponse:
    """Complete search response with analysis"""
    query: SearchQuery
    results: List[SearchResult]
    total_results: int
    search_time: float
    
    # Analysis
    query_understanding: Dict[str, Any] = field(default_factory=dict)
    search_strategy: Dict[str, Any] = field(default_factory=dict)
    
    # Enhancements
    suggested_queries: List[str] = field(default_factory=list)
    related_entities: List[Dict[str, Any]] = field(default_factory=list)
    concept_clusters: List[Dict[str, Any]] = field(default_factory=list)
    
    # Facets
    facets: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Debug information
    debug_info: Optional[Dict[str, Any]] = None


class EnhancedSearchSystem:
    """
    Production-ready enhanced search system with multi-collection support,
    knowledge graph integration, and intelligent result ranking.
    """
    
    def __init__(self):
        # Core components
        self.client = None
        self.knowledge_graph = None
        
        # Search components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Configuration
        self.config = {
            # Search parameters
            "max_results_per_collection": 100,
            "min_relevance_score": 0.3,
            "context_window_size": 200,
            
            # Knowledge graph
            "kg_entity_boost": 0.2,
            "kg_relationship_boost": 0.1,
            "kg_max_hops": 2,
            
            # Query expansion
            "max_query_expansions": 5,
            "expansion_weight": 0.3,
            
            # Result ranking
            "keyword_weight": 0.3,
            "semantic_weight": 0.5,
            "metadata_weight": 0.2,
            
            # Performance
            "enable_caching": True,
            "cache_ttl": 300,  # 5 minutes
            "parallel_search": True
        }
        
        # Cache
        self.query_cache = {}
        self.embedding_cache = {}
        
        logger.info("Enhanced Search System initialized")
    
    async def initialize(self):
        """Initialize search system components"""
        logger.info("Initializing Enhanced Search System")
        
        # Initialize components
        self.client = chromadb.PersistentClient(path="./chroma_docling_db")
        
        self.knowledge_graph = get_knowledge_graph()
        
        logger.info("Enhanced Search System initialized successfully")
    
    async def search(
        self,
        query: str,
        search_type: str = "hybrid",
        collections: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        **kwargs
    ) -> SearchResponse:
        """
        Perform enhanced search across collections
        
        Args:
            query: Search query string
            search_type: Type of search (keyword, semantic, hybrid)
            collections: List of collections to search (None = all)
            filters: Additional filters to apply
            limit: Maximum results to return
            offset: Offset for pagination
            **kwargs: Additional search options
            
        Returns:
            SearchResponse with results and analysis
        """
        start_time = datetime.now()
        
        try:
            # Create search query object
            search_query = await self._create_search_query(
                query, search_type, collections, filters, limit, offset, **kwargs
            )
            
            # Check cache
            cache_key = self._get_cache_key(search_query)
            if self.config["enable_caching"] and cache_key in self.query_cache:
                cached_response = self.query_cache[cache_key]
                cached_response.search_time = 0.001  # Indicate cached result
                return cached_response
            
            # Analyze query
            query_understanding = await self._analyze_query(search_query)
            search_query.intent = query_understanding.get("intent")
            search_query.entities = query_understanding.get("entities", [])
            search_query.concepts = query_understanding.get("concepts", [])
            
            # Expand query if enabled
            if search_query.expand_query:
                search_query = await self._expand_query(search_query)
            
            # Determine search strategy
            search_strategy = self._determine_search_strategy(search_query)
            
            # Execute search
            raw_results = await self._execute_search(search_query, search_strategy)
            
            # Enhance results with knowledge graph
            if search_query.use_knowledge_graph:
                raw_results = await self._enhance_with_knowledge_graph(
                    raw_results, search_query
                )
            
            # Rank results
            ranked_results = await self._rank_results(raw_results, search_query)
            
            # Add context if requested
            if search_query.include_context:
                ranked_results = await self._add_context(ranked_results)
            
            # Generate facets
            facets = self._generate_facets(ranked_results)
            
            # Generate suggestions
            suggested_queries = await self._generate_query_suggestions(
                search_query, ranked_results
            )
            
            # Get related entities from knowledge graph
            related_entities = await self._get_related_entities(search_query)
            
            # Get concept clusters
            concept_clusters = await self._get_concept_clusters(search_query)
            
            # Apply pagination
            paginated_results = ranked_results[offset:offset + limit]
            
            # Calculate search time
            search_time = (datetime.now() - start_time).total_seconds()
            
            # Create response
            response = SearchResponse(
                query=search_query,
                results=paginated_results,
                total_results=len(ranked_results),
                search_time=search_time,
                query_understanding=query_understanding,
                search_strategy=search_strategy,
                suggested_queries=suggested_queries,
                related_entities=related_entities,
                concept_clusters=concept_clusters,
                facets=facets
            )
            
            # Cache response
            if self.config["enable_caching"]:
                self.query_cache[cache_key] = response
                # Clean old cache entries
                asyncio.create_task(self._clean_cache())
            
            logger.info(f"Search completed: {len(paginated_results)} results in {search_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def store_content(
        self,
        content: str,
        content_type: str,
        metadata: Dict[str, Any],
        collection_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Store content in ChromaDB with embeddings
        
        Args:
            content: Text content to store
            content_type: Type of content (message, document_chunk, etc.)
            metadata: Metadata to associate with content
            collection_name: Collection to store in
            
        Returns:
            Result dictionary with success status
        """
        try:
            # Get or create collection
            collection = self.client.get_or_create_collection(name=collection_name)
            
            # Generate embedding
            embedding = await self._get_embedding(content)
            
            # Create document ID
            msg_id = metadata.get('message_id', metadata.get('document_id', str(hash(content))))
            doc_id = f"{content_type}_{msg_id}"
            
            # Clean metadata for ChromaDB compatibility (only primitive types allowed)
            cleaned_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, list):
                    # Convert lists to comma-separated strings
                    cleaned_metadata[key] = ','.join(str(item) for item in value) if value else ''
                elif isinstance(value, dict):
                    # Convert dicts to JSON strings
                    import json
                    cleaned_metadata[key] = json.dumps(value)
                elif value is None:
                    # Convert None to empty string
                    cleaned_metadata[key] = ''
                else:
                    # Keep primitive types as-is
                    cleaned_metadata[key] = str(value)
            
            # Store in ChromaDB
            collection.add(
                documents=[content],
                embeddings=[embedding.tolist()],
                metadatas=[cleaned_metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Stored content in {collection_name}: {doc_id}")
            
            return {
                "success": True,
                "document_id": doc_id,
                "collection": collection_name,
                "content_length": len(content)
            }
            
        except Exception as e:
            logger.error(f"Failed to store content: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_multiple_collections(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        search_type: str = "hybrid",
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_knowledge_graph: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search across multiple collections and return simplified results
        
        Args:
            query: Search query
            collections: List of collections to search
            search_type: Type of search
            limit: Maximum results
            filters: Additional filters
            include_knowledge_graph: Whether to include KG enhancement
            
        Returns:
            List of search result dictionaries
        """
        try:
            # Use the main search method
            response = await self.search(
                query=query,
                search_type=search_type,
                collections=collections,
                filters=filters,
                limit=limit,
                **kwargs
            )
            
            # Convert to simple format for compatibility
            results = []
            for result in response.results:
                results.append({
                    "content": result.content,
                    "score": result.score,
                    "metadata": result.metadata,
                    "document_id": result.document_id,
                    "collection": result.collection
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Multi-collection search failed: {e}")
            return []
    
    async def _create_search_query(
        self,
        query: str,
        search_type: str,
        collections: Optional[List[str]],
        filters: Optional[Dict[str, Any]],
        limit: int,
        offset: int,
        **kwargs
    ) -> SearchQuery:
        """Create enhanced search query object"""
        # Normalize query
        normalized_query = query.strip().lower()
        
        # Get all collections if not specified
        if not collections:
            try:
                collections = [col.name for col in self.client.list_collections()]
            except Exception as e:
                logger.warning(f"Failed to list collections: {e}")
                collections = []  # Default to empty list if collection listing fails
        
        return SearchQuery(
            raw_query=query,
            normalized_query=normalized_query,
            search_type=search_type,
            collections=collections,
            filters=filters or {},
            limit=limit,
            offset=offset,
            use_knowledge_graph=kwargs.get("use_knowledge_graph", True),
            expand_query=kwargs.get("expand_query", True),
            include_context=kwargs.get("include_context", True),
            min_relevance_score=kwargs.get("min_relevance_score", self.config["min_relevance_score"])
        )
    
    async def _analyze_query(self, search_query: SearchQuery) -> Dict[str, Any]:
        """Analyze query for understanding"""
        # Simple query analysis (placeholder for more advanced NLP)
        return {
            "intent": "search",
            "entities": [],
            "concepts": [],
            "temporal_context": None
        }
    
    async def _expand_query(self, search_query: SearchQuery) -> SearchQuery:
        """Expand query with related terms and entities"""
        expanded_terms = []
        
        # Expand with knowledge graph entities
        if search_query.entities:
            for entity in search_query.entities[:3]:  # Limit expansion
                related = await self.knowledge_graph.find_related_entities(
                    entity,
                    max_hops=1,
                    limit=3
                )
                for related_entity, score in related:
                    if score > 0.7:
                        expanded_terms.append(related_entity.text)
        
        # Add expanded terms to query
        if expanded_terms:
            search_query.normalized_query = f"{search_query.normalized_query} {' '.join(expanded_terms)}"
        
        return search_query
    
    def _determine_search_strategy(self, search_query: SearchQuery) -> Dict[str, Any]:
        """Determine optimal search strategy"""
        strategy = {
            "primary_method": search_query.search_type,
            "use_embeddings": search_query.search_type in ["semantic", "hybrid"],
            "use_keywords": search_query.search_type in ["keyword", "hybrid"],
            "parallel_search": self.config["parallel_search"] and len(search_query.collections) > 1,
            "optimization_hints": []
        }
        
        # Add optimization hints
        if len(search_query.entities) > 0:
            strategy["optimization_hints"].append("entity_filtering")
        
        if search_query.temporal_context:
            strategy["optimization_hints"].append("temporal_filtering")
        
        return strategy
    
    async def _execute_search(
        self,
        search_query: SearchQuery,
        search_strategy: Dict[str, Any]
    ) -> List[SearchResult]:
        """Execute search across collections"""
        all_results = []
        
        # Get query embedding if needed
        query_embedding = None
        if search_strategy["use_embeddings"]:
            query_embedding = await self._get_embedding(search_query.raw_query)
        
        # Search each collection
        if search_strategy["parallel_search"]:
            # Parallel search
            search_tasks = []
            for collection in search_query.collections:
                task = self._search_collection(
                    collection,
                    search_query,
                    query_embedding,
                    search_strategy
                )
                search_tasks.append(task)
            
            collection_results = await asyncio.gather(*search_tasks)
            for results in collection_results:
                all_results.extend(results)
        else:
            # Sequential search
            for collection in search_query.collections:
                results = await self._search_collection(
                    collection,
                    search_query,
                    query_embedding,
                    search_strategy
                )
                all_results.extend(results)
        
        return all_results
    
    async def _search_collection(
        self,
        collection_name: str,
        search_query: SearchQuery,
        query_embedding: Optional[np.ndarray],
        search_strategy: Dict[str, Any]
    ) -> List[SearchResult]:
        """Search a specific collection"""
        results = []
        
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Build search parameters
            search_params = {
                "n_results": self.config["max_results_per_collection"]
            }
            
            # Add filters
            where_clause = {}
            if search_query.filters:
                where_clause.update(search_query.filters)
            
            if search_query.entities:
                # Filter by entities
                where_clause["$or"] = [
                    {"entities": {"$contains": entity}}
                    for entity in search_query.entities
                ]
            
            if where_clause:
                search_params["where"] = where_clause
            
            # Execute search based on type
            if search_strategy["use_embeddings"] and query_embedding is not None:
                # Semantic search
                chroma_results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    **search_params
                )
            elif search_strategy["use_keywords"]:
                # Keyword search
                chroma_results = collection.query(
                    query_texts=[search_query.raw_query],
                    **search_params
                )
            else:
                # Fallback to metadata search
                chroma_results = collection.get(
                    where=where_clause,
                    limit=search_params["n_results"]
                )
            
            # Convert to SearchResult objects
            if "ids" in chroma_results:
                for i, result_id in enumerate(chroma_results["ids"][0]):
                    # Extract result data
                    content = chroma_results["documents"][0][i]
                    metadata = chroma_results["metadatas"][0][i]
                    distance = chroma_results["distances"][0][i] if "distances" in chroma_results else 0
                    
                    # Calculate relevance score
                    score = 1.0 - (distance / 2.0) if distance else 0.5
                    
                    # Create search result
                    result = SearchResult(
                        result_id=result_id,
                        score=score,
                        content=content,
                        content_type=metadata.get("content_type", "unknown"),
                        collection=collection_name,
                        document_id=metadata.get("document_id", ""),
                        document_title=metadata.get("document_title"),
                        section_title=metadata.get("section_title"),
                        chunk_index=metadata.get("chunk_index"),
                        metadata=metadata,
                        semantic_similarity=score if search_strategy["use_embeddings"] else 0
                    )
                    
                    results.append(result)
            
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {e}")
        
        return results
    
    async def _enhance_with_knowledge_graph(
        self,
        results: List[SearchResult],
        search_query: SearchQuery
    ) -> List[SearchResult]:
        """Enhance results with knowledge graph information"""
        # Get related entities from knowledge graph
        kg_entities = set()
        
        # Add query entities
        kg_entities.update(search_query.entities)
        
        # Find related entities
        for entity in search_query.entities[:3]:  # Limit for performance
            related = await self.knowledge_graph.find_related_entities(
                entity,
                max_hops=self.config["kg_max_hops"],
                limit=10
            )
            for related_entity, score in related:
                if score > 0.5:
                    kg_entities.add(related_entity.text)
        
        # Boost results containing knowledge graph entities
        for result in results:
            # Check for entity matches
            content_lower = result.content.lower()
            entity_matches = []
            kg_boost = 0.0
            
            for entity in kg_entities:
                if entity.lower() in content_lower:
                    entity_matches.append(entity)
                    kg_boost += self.config["kg_entity_boost"]
            
            result.entity_matches = entity_matches
            result.knowledge_graph_boost = min(kg_boost, 0.5)  # Cap boost
            
            # Update score
            result.score += result.knowledge_graph_boost
        
        return results
    
    async def _rank_results(
        self,
        results: List[SearchResult],
        search_query: SearchQuery
    ) -> List[SearchResult]:
        """Rank results using advanced ranking algorithm"""
        # Simple ranking by score (results are already sorted by ChromaDB)
        ranked_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        # Filter by minimum relevance score
        filtered_results = [
            r for r in ranked_results
            if r.score >= search_query.min_relevance_score
        ]
        
        return filtered_results
    
    async def _add_context(self, results: List[SearchResult]) -> List[SearchResult]:
        """Add context to search results"""
        for result in results:
            # For now, use the content as context (context reconstruction not implemented)
            context = {"full": result.content, "before": "", "after": ""}
            
            if context:
                result.context_before = context.get("before", "")
                result.context_after = context.get("after", "")
                result.full_context = context.get("full", result.content)
            
            # Highlight matches in content
            result.highlighted_content = self._highlight_matches(
                result.content,
                result.keyword_matches + result.entity_matches
            )
            
            if result.full_context:
                result.highlighted_context = self._highlight_matches(
                    result.full_context,
                    result.keyword_matches + result.entity_matches
                )
        
        return results
    
    def _generate_facets(self, results: List[SearchResult]) -> Dict[str, Dict[str, int]]:
        """Generate facets from search results"""
        facets = {
            "collections": defaultdict(int),
            "document_types": defaultdict(int),
            "entities": defaultdict(int),
            "dates": defaultdict(int)
        }
        
        for result in results:
            # Collection facet
            facets["collections"][result.collection] += 1
            
            # Document type facet
            doc_type = result.metadata.get("file_type", "unknown")
            facets["document_types"][doc_type] += 1
            
            # Entity facets
            for entity in result.entity_matches:
                facets["entities"][entity] += 1
            
            # Date facets (simplified)
            if "processed_at" in result.metadata:
                date_str = result.metadata["processed_at"][:10]  # YYYY-MM-DD
                facets["dates"][date_str] += 1
        
        # Convert defaultdicts to regular dicts
        return {k: dict(v) for k, v in facets.items()}
    
    async def _generate_query_suggestions(
        self,
        search_query: SearchQuery,
        results: List[SearchResult]
    ) -> List[str]:
        """Generate suggested queries based on results"""
        suggestions = []
        
        # Entity-based suggestions
        top_entities = self._get_top_entities_from_results(results, limit=5)
        for entity in top_entities:
            if entity not in search_query.raw_query:
                suggestions.append(f"{search_query.raw_query} {entity}")
        
        # Concept-based suggestions
        if search_query.concepts:
            for concept in search_query.concepts[:3]:
                suggestions.append(f"{concept} related to {search_query.raw_query}")
        
        # Refinement suggestions
        if len(results) > 50:
            suggestions.append(f"{search_query.raw_query} recent")
            suggestions.append(f"{search_query.raw_query} summary")
        
        return suggestions[:5]  # Limit suggestions
    
    async def _get_related_entities(self, search_query: SearchQuery) -> List[Dict[str, Any]]:
        """Get related entities from knowledge graph"""
        related_entities = []
        
        for entity in search_query.entities[:3]:  # Limit for performance
            related = await self.knowledge_graph.find_related_entities(
                entity,
                max_hops=2,
                limit=5
            )
            
            for related_entity, score in related:
                related_entities.append({
                    "entity": related_entity.text,
                    "type": related_entity.entity_type,
                    "related_to": entity,
                    "relevance": score,
                    "frequency": related_entity.frequency
                })
        
        return related_entities
    
    async def _get_concept_clusters(self, search_query: SearchQuery) -> List[Dict[str, Any]]:
        """Get concept clusters related to query"""
        clusters = await self.knowledge_graph.find_concept_clusters(
            search_query.entities + search_query.concepts
        )
        
        cluster_info = []
        for cluster in clusters[:5]:  # Limit clusters
            cluster_info.append({
                "cluster_id": cluster.cluster_id,
                "name": cluster.name,
                "keywords": cluster.keywords[:10],
                "coherence": cluster.coherence_score,
                "size": len(cluster.entity_ids)
            })
        
        return cluster_info
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching"""
        # Check cache
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Generate embedding
        embedding = await asyncio.to_thread(
            self.embedding_model.encode,
            text,
            normalize_embeddings=True
        )
        
        # Cache embedding
        self.embedding_cache[text] = embedding
        
        # Limit cache size
        if len(self.embedding_cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self.embedding_cache.keys())[:100]
            for key in oldest_keys:
                del self.embedding_cache[key]
        
        return embedding
    
    def _highlight_matches(self, text: str, matches: List[str]) -> str:
        """Highlight matching terms in text"""
        highlighted = text
        
        for match in matches:
            # Case-insensitive highlighting
            import re
            pattern = re.compile(re.escape(match), re.IGNORECASE)
            highlighted = pattern.sub(f"**{match}**", highlighted)
        
        return highlighted
    
    def _get_top_entities_from_results(
        self,
        results: List[SearchResult],
        limit: int = 5
    ) -> List[str]:
        """Extract top entities from search results"""
        entity_counts = defaultdict(int)
        
        for result in results:
            for entity in result.entity_matches:
                entity_counts[entity] += 1
        
        # Sort by frequency
        sorted_entities = sorted(
            entity_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [entity for entity, _ in sorted_entities[:limit]]
    
    def _get_cache_key(self, search_query: SearchQuery) -> str:
        """Generate cache key for search query"""
        key_parts = [
            search_query.normalized_query,
            search_query.search_type,
            ",".join(sorted(search_query.collections)),
            str(search_query.filters),
            str(search_query.limit),
            str(search_query.offset)
        ]
        return "|".join(key_parts)
    
    async def _clean_cache(self):
        """Clean expired cache entries"""
        # This is simplified - in production would use TTL
        if len(self.query_cache) > 100:
            # Remove oldest half
            keys_to_remove = list(self.query_cache.keys())[:50]
            for key in keys_to_remove:
                del self.query_cache[key]
    
    # Additional methods needed by the API
    
    async def store_document(self, result) -> Dict[str, Any]:
        """Store a document processing result"""
        try:
            # Get or create collection for documents
            collection = self.client.get_or_create_collection(name="documents")
            
            # Store each chunk
            for i, chunk in enumerate(result.chunks):
                # Generate embedding
                embedding = await self._get_embedding(chunk.content)
                
                # Create document ID
                doc_id = f"{result.document_id}_chunk_{i}"
                
                # Prepare metadata
                metadata = {
                    "document_id": result.document_id,
                    "filename": result.filename,
                    "chunk_index": i,
                    "content_type": "document_chunk",
                    "file_type": getattr(result, 'file_type', 'unknown'),
                    "processed_at": datetime.now().isoformat()
                }
                
                # Add to collection
                collection.add(
                    documents=[chunk.content],
                    embeddings=[embedding.tolist()],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
            
            return {
                "success": True,
                "document_id": result.document_id,
                "chunks_stored": len(result.chunks)
            }
            
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document information"""
        try:
            # Search for document chunks
            collection = self.client.get_collection(name="documents")
            results = collection.get(
                where={"document_id": document_id},
                limit=1
            )
            
            if results and results.get("ids"):
                metadata = results["metadatas"][0]
                return {
                    "document_id": document_id,
                    "filename": metadata.get("filename", "Unknown"),
                    "file_type": metadata.get("file_type", "unknown"),
                    "processed_at": metadata.get("processed_at")
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document info: {e}")
            return None
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks"""
        try:
            collection = self.client.get_collection(name="documents")
            
            # Get all chunks for this document
            results = collection.get(
                where={"document_id": document_id}
            )
            
            if results and results.get("ids"):
                # Delete all chunks
                collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False


# Global instance
_search_system = None


def get_enhanced_search_system() -> EnhancedSearchSystem:
    """Get global enhanced search system instance"""
    global _search_system
    if _search_system is None:
        _search_system = EnhancedSearchSystem()
    return _search_system


# Test function
if __name__ == "__main__":
    async def test_search():
        search_system = EnhancedSearchSystem()
        await search_system.initialize()
        
        # Test search
        response = await search_system.search(
            query="machine learning algorithms",
            search_type="hybrid",
            limit=5,
            use_knowledge_graph=True,
            expand_query=True
        )
        
        print(f"Search Results: {len(response.results)} found")
        print(f"Total Results: {response.total_results}")
        print(f"Search Time: {response.search_time:.3f}s")
        
        for i, result in enumerate(response.results):
            print(f"\n{i+1}. Score: {result.score:.3f}")
            print(f"   Content: {result.content[:100]}...")
            print(f"   Collection: {result.collection}")
            print(f"   Entity Matches: {result.entity_matches}")
        
        print(f"\nSuggested Queries: {response.suggested_queries}")
        print(f"Related Entities: {len(response.related_entities)}")
        print(f"Concept Clusters: {len(response.concept_clusters)}")
    
    # Run test
    asyncio.run(test_search())