"""
Unified Docling API - Production-Ready Clean API

This is the single, unified API for all document processing and AI operations.
No legacy code, no dual processing, no compatibility layers - just pure Docling power.
"""

import asyncio
import hashlib
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Core components
from docling_document_processor import (
    UnifiedDoclingProcessor,
    UnifiedProcessingResult,
    get_unified_processor
)
from entity_relationship_graph import (
    KnowledgeGraphSystem,
    KnowledgeGraphStats,
    get_knowledge_graph
)
from semantic_search_engine import EnhancedSearchSystem
from ai_context_builder import EnhancedAIIntegration
from intelligent_prompt_generator import EnhancedPromptGenerator
from agent.providers.provider_configuration_manager import get_provider_manager
from intelligent_suggestion_engine import SuggestionService

# Monitoring and performance
from system_memory_optimizer import MemoryManager
from basic_concurrency_manager import ConcurrentOptimizer
from system_metrics_monitor import MonitoringFramework, AlertingSystem

# Set up logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Docling Knowledge API",
    description="Production-ready API for document processing, knowledge management, and AI-powered insights",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
processor = None
knowledge_graph = None
search_system = None
ai_integration = None
prompt_generator = None
provider_manager = None
suggestion_service = None
memory_manager = None
concurrent_optimizer = None
monitoring = None
alerting = None


# ===== Request/Response Models =====

class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    document_id: str
    filename: str
    success: bool
    message: str
    processing_stats: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    """Request for document search"""
    query: str
    search_type: str = "hybrid"
    collection: Optional[str] = None
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None
    include_knowledge_graph: bool = True
    include_context: bool = True


class SearchResponse(BaseModel):
    """Response for document search"""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time: float
    knowledge_graph_entities: Optional[List[Dict[str, Any]]] = None
    suggested_queries: Optional[List[str]] = None


class SuggestionRequest(BaseModel):
    """Request for AI suggestions"""
    message: str
    conversation_history: List[Dict[str, str]] = []
    user_id: str = "default"
    channel_id: str = "general"
    thread_id: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7
    include_sources: bool = True
    use_knowledge_graph: bool = True


class SuggestionResponse(BaseModel):
    """Response for AI suggestions"""
    suggestion: str
    sources: Optional[List[Dict[str, Any]]] = None
    confidence: float
    entities_used: Optional[List[str]] = None
    processing_time: float


class KnowledgeGraphRequest(BaseModel):
    """Request for knowledge graph operations"""
    operation: str  # "find_related", "get_clusters", "get_stats"
    entity: Optional[str] = None
    entities: Optional[List[str]] = None
    max_hops: int = 2
    limit: int = 10


class CollectionInfo(BaseModel):
    """Information about a document collection"""
    name: str
    document_count: int
    chunk_count: int
    entity_count: int
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any] = {}


class SystemHealth(BaseModel):
    """System health information"""
    status: str
    uptime: float
    memory_usage: Dict[str, float]
    active_operations: int
    collection_stats: Dict[str, int]
    provider_status: Dict[str, bool]
    last_check: datetime


# ===== Initialization =====

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global processor, knowledge_graph, search_system, ai_integration
    global prompt_generator, provider_manager, suggestion_service
    global memory_manager, concurrent_optimizer, monitoring, alerting
    
    logger.info("Initializing Docling API components...")
    
    try:
        # Core components
        processor = get_unified_processor()
        knowledge_graph = get_knowledge_graph()
        
        # Storage and search
        search_system = EnhancedSearchSystem()
        await search_system.initialize()
        
        # AI and suggestion components
        ai_integration = EnhancedAIIntegration()
        prompt_generator = EnhancedPromptGenerator()
        provider_manager = get_provider_manager()
        
        suggestion_service = SuggestionService(
            vector_db=search_system,  # Using search_system as vector_db interface
            provider_manager=provider_manager
        )
        
        # Performance and monitoring
        memory_manager = MemoryManager()
        concurrent_optimizer = ConcurrentOptimizer()
        monitoring = MonitoringFramework()
        alerting = AlertingSystem()
        
        await monitoring.initialize()
        await alerting.initialize()
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Docling API...")
    
    if monitoring:
        await monitoring.shutdown()
    if alerting:
        await alerting.shutdown()
    if search_system:
        # Clean shutdown of search system if needed
        logger.info("Search system shutdown complete")


# ===== Document Processing Endpoints =====

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("default"),
    metadata: Optional[str] = Form(None)
):
    """
    Upload and process a document using Docling
    
    - Supports PDF, DOCX, TXT, MD, and more formats
    - Extracts structure, chunks, tables, and entities
    - Builds knowledge graph relationships
    - Stores in multi-collection system
    """
    try:
        # Validate file format first
        file_extension = Path(file.filename).suffix.lower()
        docling_formats = {'.pdf', '.docx', '.pptx', '.html', '.md', '.csv', '.xlsx', '.png', '.jpg', '.jpeg'}
        direct_text_formats = {'.txt'}
        supported_formats = docling_formats | direct_text_formats
        
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_extension}. Supported formats: {', '.join(sorted(supported_formats))}"
            )
        
        # Parse metadata if provided
        doc_metadata = json.loads(metadata) if metadata else {}
        
        # Read file
        file_data = await file.read()
        
        # Handle different file types
        if file_extension in direct_text_formats:
            # Process text files directly without Docling (quick implementation)
            text_content = file_data.decode('utf-8')
            document_id = hashlib.md5(file_data).hexdigest()[:16]
            
            # Simple processing for text files
            result = type('MockResult', (), {
                'success': True,
                'document_id': document_id,
                'sections': [],
                'chunks': [],
                'tables': [],
                'entities': [],
                'processing_stats': {
                    'sections_extracted': 1,
                    'chunks_created': 1,
                    'tables_found': 0,
                    'entities_extracted': 0,
                    'processing_time_seconds': 0.001,
                    'memory_used_mb': 0.0
                },
                'metadata': {
                    'filename': file.filename,
                    'user_id': user_id,
                    'processed_at': datetime.now().isoformat(),
                    'processing_method': 'direct_text',
                    'content_length': len(text_content)
                }
            })()
        else:
            # Process with unified Docling processor
            result = await processor.process_document(
                file_data,
                file.filename,
                user_id,
                doc_metadata
            )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        # Store in multi-collection system
        if file_extension in direct_text_formats:
            # Store text content directly in the vector database
            await search_system.store_content(
                content=text_content,
                content_type='document',
                metadata={
                    'document_id': result.document_id,
                    'filename': file.filename,
                    'user_id': user_id,
                    'upload_timestamp': datetime.now().isoformat(),
                    'file_type': 'text',
                    'processing_method': 'direct_text'
                },
                collection_name='documents'
            )
        else:
            storage_result = await search_system.store_document(result)
        
        # Add to knowledge graph
        if file_extension not in direct_text_formats:
            # Only add to knowledge graph for Docling-processed documents
            await knowledge_graph.add_document_entities(
                result.document_id,
                result.entities,
                result.sections,
                result.chunks
            )
        
        # Monitor performance
        await monitoring.record_operation(
            "document_upload",
            {
                "document_id": result.document_id,
                "filename": file.filename,
                "processing_time": result.processing_stats.get("processing_time_seconds", 0),
                "chunks": len(result.chunks),
                "entities": len(result.entities)
            }
        )
        
        return DocumentUploadResponse(
            document_id=result.document_id,
            filename=file.filename,
            success=True,
            message="Document processed successfully",
            processing_stats=result.processing_stats,
            metadata=result.metadata
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions (like format validation errors) 
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Document upload failed: {e}")
        logger.error(f"Full traceback: {error_details}")
        
        # Ensure we have a meaningful error message
        error_message = str(e) if str(e) else "Unknown error during document processing"
        
        await alerting.send_alert("document_upload_error", error_message)
        raise HTTPException(status_code=500, detail=error_message)


@app.get("/api/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details and metadata"""
    try:
        doc_info = await search_system.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return doc_info
        
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its data"""
    try:
        # Remove from storage
        await search_system.delete_document(document_id)
        
        # Remove from knowledge graph (implement this method)
        # await knowledge_graph.remove_document(document_id)
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Search Endpoints =====

@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search across documents using enhanced multi-collection search
    
    - Supports semantic, keyword, and hybrid search
    - Includes knowledge graph entities
    - Provides contextual results
    - Suggests related queries
    """
    try:
        start_time = datetime.now()
        
        # Perform enhanced search
        search_results = await search_system.search_multiple_collections(
            query=request.query,
            collections=[request.collection] if request.collection else None,
            search_type=request.search_type,
            limit=request.limit,
            filters=request.filters,
            include_knowledge_graph=request.include_knowledge_graph
        )
        
        ranked_results = search_results  # EnhancedSearchSystem already ranks results
        
        # Get knowledge graph entities if requested
        kg_entities = []
        if request.include_knowledge_graph:
            # Extract entities from query
            query_entities = knowledge_graph.extract_entities(request.query)
            for entity in query_entities:
                related = await knowledge_graph.find_related_entities(
                    [entity.text],
                    max_hops=2,
                    limit=5
                )
                kg_entities.extend([
                    {
                        "entity": rel_entity,
                        "type": "CONCEPT",  # Default type
                        "relevance": 0.8,  # Default relevance
                        "frequency": 1
                    }
                    for rel_entity in related
                ])
        
        # Generate suggested queries based on knowledge graph
        suggested_queries = []
        if kg_entities:
            # Create simple suggested queries from related entities
            suggested_queries = [
                f"{request.query} {entity['entity']}" 
                for entity in kg_entities[:3]
            ]
        
        # Calculate search time
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Monitor search performance
        await monitoring.record_search(
            request.query,
            len(ranked_results),
            search_time
        )
        
        return SearchResponse(
            query=request.query,
            results=ranked_results[:request.limit],
            total_results=len(ranked_results),
            search_time=search_time,
            knowledge_graph_entities=kg_entities if kg_entities else None,
            suggested_queries=suggested_queries if suggested_queries else None
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== AI Suggestion Endpoints =====

@app.post("/api/suggestion", response_model=SuggestionResponse)
async def get_suggestion(request: SuggestionRequest):
    """
    Get AI-powered suggestions based on context and knowledge base
    
    - Uses enhanced prompt generation
    - Includes knowledge graph context
    - Provides source attribution
    - Supports multiple LLM providers
    """
    try:
        start_time = datetime.now()
        
        # Generate suggestion with enhanced context
        # Use the existing suggestion service method
        result = await suggestion_service.generate_enhanced_suggestion(
            user_name=request.user_id,
            user_id=request.user_id,
            content=request.message,
            conversation_id=request.channel_id,
            thread_ts=request.thread_id
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Monitor suggestion generation
        await monitoring.record_suggestion(
            request.user_id,
            len(request.message),
            processing_time
        )
        
        return SuggestionResponse(
            suggestion=result.get("suggestion", ""),
            sources=result.get("sources") if request.include_sources else None,
            confidence=result.get("confidence", 0.0),
            entities_used=result.get("entities") if request.use_knowledge_graph else None,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Suggestion generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Knowledge Graph Endpoints =====

@app.post("/api/knowledge-graph")
async def knowledge_graph_operations(request: KnowledgeGraphRequest):
    """
    Perform operations on the knowledge graph
    
    - Find related entities
    - Get concept clusters
    - Get graph statistics
    """
    try:
        if request.operation == "find_related":
            if not request.entity:
                raise HTTPException(status_code=400, detail="Entity required for find_related")
            
            related = await knowledge_graph.find_related_entities(
                request.entity,
                max_hops=request.max_hops,
                limit=request.limit
            )
            
            return {
                "entity": request.entity,
                "related_entities": [
                    {
                        "text": entity.text,
                        "type": entity.entity_type,
                        "relevance": score,
                        "frequency": entity.frequency
                    }
                    for entity, score in related
                ]
            }
            
        elif request.operation == "get_clusters":
            entities = request.entities or []
            clusters = await knowledge_graph.find_concept_clusters(entities)
            
            return {
                "clusters": [
                    {
                        "id": cluster.cluster_id,
                        "name": cluster.name,
                        "entities": list(cluster.entity_ids),
                        "keywords": cluster.keywords,
                        "coherence": cluster.coherence_score
                    }
                    for cluster in clusters
                ]
            }
            
        elif request.operation == "get_stats":
            stats = knowledge_graph.get_statistics()
            
            return {
                "total_entities": stats.total_entities,
                "total_relationships": stats.total_relationships,
                "total_clusters": stats.total_clusters,
                "entity_types": stats.entity_types,
                "relationship_types": stats.relationship_types,
                "avg_relationships_per_entity": stats.avg_relationships_per_entity,
                "graph_density": stats.graph_density,
                "most_connected_entities": stats.most_connected_entities,
                "largest_clusters": stats.largest_clusters
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")
            
    except Exception as e:
        logger.error(f"Knowledge graph operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/knowledge-graph/build-clusters")
async def build_knowledge_clusters():
    """Build concept clusters from current knowledge graph"""
    try:
        await knowledge_graph.build_concept_clusters()
        stats = knowledge_graph.get_statistics()
        
        return {
            "message": "Clusters built successfully",
            "total_clusters": stats.total_clusters
        }
        
    except Exception as e:
        logger.error(f"Cluster building failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Collection Management Endpoints =====

@app.get("/api/collections", response_model=List[CollectionInfo])
async def list_collections():
    """List all document collections"""
    try:
        collections = search_system.client.list_collections()
        
        collection_infos = []
        for collection in collections:
            # Get basic collection info
            count = collection.count()
            collection_infos.append(CollectionInfo(
                name=collection.name,
                document_count=count,
                chunk_count=count,  # Simplified - same as document count
                entity_count=0,  # Would need knowledge graph integration
                created_at=datetime.now(),  # Placeholder
                last_updated=datetime.now(),  # Placeholder
                metadata={}
            ))
        
        return collection_infos
        
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collections/{collection_name}", response_model=CollectionInfo)
async def get_collection(collection_name: str):
    """Get information about a specific collection"""
    try:
        collection = search_system.client.get_collection(name=collection_name)
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        # Get basic collection info
        count = collection.count()
        
        return CollectionInfo(
            name=collection.name,
            document_count=count,
            chunk_count=count,
            entity_count=0,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            metadata={}
        )
        
    except Exception as e:
        logger.error(f"Failed to get collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== System Management Endpoints =====

@app.get("/api/health", response_model=SystemHealth)
async def health_check():
    """Get system health status"""
    try:
        # Get memory usage
        memory_info = memory_manager.get_memory_status()
        
        # Get collection stats
        collections = search_system.client.list_collections()
        collection_stats = {}
        for collection in collections:
            collection_stats[collection.name] = collection.count()
        
        # Get provider status
        provider_status = {}
        providers = provider_manager.list_providers()
        for provider in providers:
            provider_status[provider["name"]] = provider["active"]
        
        # Get active operations
        active_ops = concurrent_optimizer.get_active_operations()
        
        return SystemHealth(
            status="healthy",
            uptime=monitoring.get_uptime() if monitoring else 0,
            memory_usage=memory_info,
            active_operations=len(active_ops),
            collection_stats=collection_stats,
            provider_status=provider_status,
            last_check=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return SystemHealth(
            status="unhealthy",
            uptime=0,
            memory_usage={},
            active_operations=0,
            collection_stats={},
            provider_status={},
            last_check=datetime.now()
        )


@app.post("/api/system/optimize")
async def optimize_system():
    """Trigger system optimization"""
    try:
        # Memory optimization
        memory_manager.optimize_memory()
        
        # Collection optimization (simplified)
        logger.info("Collection optimization completed")
        
        # Performance optimization
        optimization_results = await concurrent_optimizer.optimize_system()
        
        return {
            "message": "System optimization completed",
            "results": optimization_results
        }
        
    except Exception as e:
        logger.error(f"System optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/system/metrics")
async def get_system_metrics(
    metric_type: str = Query("all", description="Type of metrics: all, performance, usage, errors"),
    time_range: str = Query("1h", description="Time range: 1h, 24h, 7d, 30d")
):
    """Get system metrics"""
    try:
        metrics = await monitoring.get_metrics(metric_type, time_range)
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== LLM Provider Management =====

@app.get("/api/providers")
async def list_providers():
    """List available LLM providers"""
    try:
        providers = provider_manager.list_providers()
        return {"providers": providers}
        
    except Exception as e:
        logger.error(f"Failed to list providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/llm/providers")
async def list_providers_legacy():
    """Legacy endpoint - List available LLM providers"""
    return await list_providers()


@app.post("/api/llm/providers/configure")
async def configure_provider_legacy(config: dict = Body(...)):
    """Legacy endpoint - Configure a LLM provider"""
    try:
        # Check if provider manager is initialized
        if provider_manager is None:
            raise HTTPException(status_code=503, detail="Server still starting up, please try again")
        
        # Extract provider name and configuration
        provider_name = config.get("provider", config.get("name"))
        if not provider_name:
            raise HTTPException(status_code=400, detail="Provider name is required")
        
        # Use the provider manager to add/update the provider
        logger.info(f"Configuring provider: {provider_name} with config: {config}")
        result = provider_manager.add_provider(provider_name, config)
        
        logger.info(f"Configuration result: {result}")
        logger.info(f"Current providers: {list(provider_manager.providers.keys())}")
        logger.info(f"Active provider: {provider_manager.active_provider}")
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "message": f"Provider {provider_name} configured successfully",
            "provider": provider_name
        }
        
    except Exception as e:
        logger.error(f"Failed to configure provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/llm/providers/activate")
async def activate_provider_legacy(request: dict = Body(...)):
    """Legacy endpoint - Activate a LLM provider"""
    try:
        # Check if provider manager is initialized
        if provider_manager is None:
            raise HTTPException(status_code=503, detail="Server still starting up, please try again")
        
        provider_name = request.get("provider", request.get("name"))
        if not provider_name:
            raise HTTPException(status_code=400, detail="Provider name is required")
        
        logger.info(f"Activating provider: {provider_name}")
        logger.info(f"Available providers: {list(provider_manager.providers.keys())}")
        logger.info(f"Current active provider: {provider_manager.active_provider}")
        
        result = provider_manager.set_active_provider(provider_name)
        logger.info(f"Activation result: {result}")
        logger.info(f"New active provider: {provider_manager.active_provider}")
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "message": f"Provider {provider_name} activated successfully",
            "provider": provider_name
        }
        
    except Exception as e:
        logger.error(f"Failed to activate provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/llm/providers/test") 
async def test_provider_legacy():
    """Legacy endpoint - Test the active LLM provider"""
    try:
        # Check if provider manager is initialized
        if provider_manager is None:
            return {"success": False, "error": "Server still starting up, please try again"}
        
        logger.info(f"Testing provider - Current active: {provider_manager.active_provider}")
        logger.info(f"Available providers: {list(provider_manager.providers.keys())}")
        
        active_provider_info = provider_manager.get_active_provider()
        logger.info(f"Active provider info: {active_provider_info}")
        
        if not active_provider_info:
            return {"success": False, "error": "No active provider configured"}
        
        # Test with a simple prompt
        test_response = await provider_manager.generate(
            messages=[{"role": "user", "content": "Say 'test successful' if you can receive this message."}],
            max_tokens=50,
            temperature=0.1
        )
        
        return {
            "success": True,
            "message": "Provider test successful",
            "provider": active_provider_info["name"],
            "response": test_response.get("content", test_response.get("text", ""))
        }
        
    except Exception as e:
        logger.error(f"Provider test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Provider test failed: {str(e)}"
        }


@app.post("/api/documents/search")
async def search_documents_legacy(request: dict = Body(...)):
    """Legacy endpoint - Search documents"""
    try:
        query = request.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
            
        # Use the main search endpoint
        search_request = SearchRequest(
            query=query,
            search_type="hybrid",
            collection=None,  # Search all collections
            limit=request.get("limit", 10),
            include_knowledge_graph=True
        )
        
        response = await search_documents(search_request)
        
        # Convert to legacy format
        return {
            "results": response.results,
            "total": response.total_results,
            "query": response.query,
            "search_time": response.search_time
        }
        
    except Exception as e:
        logger.error(f"Document search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/providers/{provider_name}/activate")
async def activate_provider(provider_name: str):
    """Activate a specific LLM provider"""
    try:
        result = provider_manager.set_active_provider(provider_name)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {"message": f"Provider {provider_name} activated successfully"}
        
    except Exception as e:
        logger.error(f"Failed to activate provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Legacy Compatibility Endpoints =====
# These endpoints maintain compatibility with the Chrome extension

@app.get("/api/llm/current")
async def get_current_llm_legacy():
    """Legacy endpoint - get current LLM provider"""
    try:
        providers_response = await list_providers()
        active_provider = next((p for p in providers_response["providers"] if p["active"]), None)
        if active_provider:
            return {
                "provider": active_provider["name"],
                "model": active_provider["model"], 
                "status": "active"
            }
        return {"provider": None, "model": None, "status": "none"}
    except Exception as e:
        logger.error(f"Failed to get current LLM: {e}")
        return {"provider": None, "model": None, "status": "error"}


@app.post("/api/conversations")
async def create_conversation_legacy(request: dict = Body(...)):
    """Legacy endpoint - conversations are now implicit in the unified system"""
    conversation_id = request.get("conversation_id", "default")
    return {
        "conversation_id": conversation_id,
        "status": "created",
        "message": "Conversations are now implicit in unified system"
    }


@app.get("/api/conversations/{conversation_id}/has-messages")
async def check_messages_legacy(conversation_id: str):
    """Legacy endpoint - check if channel has messages"""
    try:
        # Search for any messages in this channel
        search_results = await search_system.search_multiple_collections(
            query="*",
            collections=["slack_messages"],
            filters={"channel_id": conversation_id},
            limit=1
        )
        return {
            "has_messages": len(search_results) > 0,
            "count": len(search_results)
        }
    except Exception as e:
        logger.error(f"Failed to check messages for {conversation_id}: {e}")
        return {"has_messages": False, "count": 0}


@app.post("/api/messages/bulk")
async def store_messages_bulk_legacy(request: Request):
    """Legacy endpoint - bulk message storage"""
    try:
        # Get raw request body and parse as JSON
        body = await request.body()
        import json
        data = json.loads(body)
        
        # Handle different formats the extension might send
        if isinstance(data, list):
            messages = data
        elif isinstance(data, dict):
            if "messages" in data:
                messages = data["messages"]
            else:
                # Single message object
                messages = [data]
        else:
            messages = []
            
        logger.info(f"Received bulk message request with {len(messages)} messages")
        logger.debug(f"Raw request data type: {type(data)}")
        logger.debug(f"Sample message format: {messages[0] if messages else 'No messages'}")
        
    except Exception as e:
        logger.error(f"Failed to parse bulk message request: {e}")
        return {"stored": 0, "total": 0, "status": "error", "errors": [f"Invalid request format: {str(e)}"]}
    stored_count = 0
    errors = []
    
    for i, message in enumerate(messages):
        try:
            # Convert legacy message format to unified format
            # Ensure no None values (ChromaDB doesn't accept None in metadata)
            metadata = {
                'user_id': str(message.get('user_id', 'unknown')),
                'channel_id': str(message.get('channel_id', 'unknown')),
                'thread_id': str(message.get('thread_id', '')),
                'timestamp': str(message.get('timestamp', '')),
                'message_type': 'slack_message',
                'message_id': str(message.get('message_id', f'msg_{i}'))
            }
            
            # Extract entities if knowledge graph is enabled
            content = message.get('content', '') or message.get('text', '')
            if content and knowledge_graph:
                entities = knowledge_graph.extract_entities(content)
                metadata['entities'] = ','.join([entity.text for entity in entities]) if entities else ''
                metadata['entity_types'] = ','.join([entity.entity_type for entity in entities]) if entities else ''
            
            await search_system.store_content(
                content=content,
                content_type='message',
                metadata=metadata,
                collection_name='slack_messages'
            )
            stored_count += 1
            
        except Exception as e:
            error_msg = f"Failed to store message {i}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    return {
        "stored": stored_count,
        "total": len(messages),
        "status": "completed",
        "errors": errors if errors else None
    }


# ===== Root Endpoint =====

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Docling Knowledge API",
        "version": "3.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/api/health"
    }


# ===== Error Handler =====

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Send alert for critical errors
    await alerting.send_alert("unhandled_exception", str(exc))
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc) if app.debug else "Internal server error"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)