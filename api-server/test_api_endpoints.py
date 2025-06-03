"""
Comprehensive Unit Tests for All API Endpoints
Testing the Docling-powered FastAPI backend endpoints
"""

import pytest
import asyncio
import json
import io
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile

# Import the FastAPI app
from api_server import app

# Create test client
client = TestClient(app)


class TestDocumentEndpoints:
    """Test document processing endpoints"""
    
    @patch('api_server.processor')
    @patch('api_server.search_system')
    @patch('api_server.knowledge_graph')
    @patch('api_server.monitoring')
    def test_upload_document_success(self, mock_monitoring, mock_kg, mock_search, mock_processor):
        """Test successful document upload"""
        # Mock processor response
        mock_result = Mock()
        mock_result.success = True
        mock_result.document_id = "test-doc-123"
        mock_result.entities = []
        mock_result.sections = []
        mock_result.chunks = []
        mock_result.processing_stats = {"processing_time_seconds": 1.5}
        mock_result.metadata = {"filename": "test.pdf"}
        mock_result.error = None
        
        mock_processor.process_document = AsyncMock(return_value=mock_result)
        mock_search.store_document = AsyncMock()
        mock_kg.add_document_entities = AsyncMock()
        mock_monitoring.record_operation = AsyncMock()
        
        # Create test file
        test_file = ("test.pdf", b"test pdf content", "application/pdf")
        
        response = client.post(
            "/api/documents/upload",
            files={"file": test_file},
            data={"user_id": "test-user", "metadata": '{"source": "test"}'}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["document_id"] == "test-doc-123"
        assert data["filename"] == "test.pdf"
        assert "processing_stats" in data
    
    @patch('api_server.processor')
    def test_upload_document_failure(self, mock_processor):
        """Test document upload failure"""
        # Mock processor failure
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Failed to process document"
        
        mock_processor.process_document = AsyncMock(return_value=mock_result)
        
        test_file = ("test.pdf", b"test pdf content", "application/pdf")
        
        response = client.post(
            "/api/documents/upload",
            files={"file": test_file},
            data={"user_id": "test-user"}
        )
        
        assert response.status_code == 400
        assert "Failed to process document" in response.json()["detail"]
    
    @patch('api_server.search_system')
    def test_get_document_success(self, mock_search):
        """Test successful document retrieval"""
        mock_search.get_document_info = AsyncMock(return_value={
            "document_id": "test-doc-123",
            "filename": "test.pdf",
            "metadata": {"source": "test"}
        })
        
        response = client.get("/api/documents/test-doc-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "test-doc-123"
        assert data["filename"] == "test.pdf"
    
    @patch('api_server.search_system')
    def test_get_document_not_found(self, mock_search):
        """Test document not found"""
        mock_search.get_document_info = AsyncMock(return_value=None)
        
        response = client.get("/api/documents/nonexistent")
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]
    
    @patch('api_server.search_system')
    def test_delete_document_success(self, mock_search):
        """Test successful document deletion"""
        mock_search.delete_document = AsyncMock()
        
        response = client.delete("/api/documents/test-doc-123")
        
        assert response.status_code == 200
        data = response.json()
        assert "Document deleted successfully" in data["message"]


class TestSearchEndpoints:
    """Test search endpoints"""
    
    @patch('api_server.search_system')
    @patch('api_server.knowledge_graph')
    @patch('api_server.monitoring')
    def test_search_documents_success(self, mock_monitoring, mock_kg, mock_search):
        """Test successful document search"""
        # Mock search results
        mock_search.search_multiple_collections = AsyncMock(return_value=[
            {
                "id": "doc1",
                "content": "Test content 1",
                "metadata": {"source": "test1"},
                "score": 0.9
            },
            {
                "id": "doc2", 
                "content": "Test content 2",
                "metadata": {"source": "test2"},
                "score": 0.8
            }
        ])
        
        # Mock knowledge graph
        mock_entity = Mock()
        mock_entity.text = "test entity"
        mock_kg.extract_entities = Mock(return_value=[mock_entity])
        mock_kg.find_related_entities = AsyncMock(return_value=["related1", "related2"])
        mock_monitoring.record_search = AsyncMock()
        
        search_request = {
            "query": "test query",
            "search_type": "hybrid",
            "limit": 10,
            "include_knowledge_graph": True
        }
        
        response = client.post("/api/search", json=search_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert len(data["results"]) == 2
        assert data["total_results"] == 2
        assert "search_time" in data
    
    @patch('api_server.search_system')
    def test_search_documents_minimal(self, mock_search):
        """Test search with minimal parameters"""
        mock_search.search_multiple_collections = AsyncMock(return_value=[])
        
        search_request = {"query": "simple query"}
        
        response = client.post("/api/search", json=search_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "simple query"
        assert data["results"] == []
    
    @patch('api_server.search_system')
    def test_search_documents_legacy(self, mock_search):
        """Test legacy search endpoint"""
        mock_search.search_multiple_collections = AsyncMock(return_value=[
            {"id": "doc1", "content": "Test", "score": 0.9}
        ])
        
        request_data = {"query": "legacy search", "limit": 5}
        
        response = client.post("/api/documents/search", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "legacy search"
        assert len(data["results"]) == 1


class TestSuggestionEndpoints:
    """Test AI suggestion endpoints"""
    
    @patch('api_server.suggestion_service')
    @patch('api_server.monitoring')
    def test_get_suggestion_success(self, mock_monitoring, mock_suggestion):
        """Test successful suggestion generation"""
        mock_suggestion.generate_enhanced_suggestion = AsyncMock(return_value={
            "suggestion": "This is a test suggestion",
            "sources": [{"title": "Test Doc", "url": "/test"}],
            "confidence": 0.85,
            "entities": ["test", "suggestion"]
        })
        mock_monitoring.record_suggestion = AsyncMock()
        
        suggestion_request = {
            "message": "Help me write a response",
            "user_id": "test-user",
            "channel_id": "test-channel",
            "include_sources": True,
            "use_knowledge_graph": True
        }
        
        response = client.post("/api/suggestion", json=suggestion_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["suggestion"] == "This is a test suggestion"
        assert data["confidence"] == 0.85
        assert "processing_time" in data
        assert "sources" in data
        assert "entities_used" in data
    
    @patch('api_server.suggestion_service')
    def test_get_suggestion_minimal(self, mock_suggestion):
        """Test suggestion with minimal parameters"""
        mock_suggestion.generate_enhanced_suggestion = AsyncMock(return_value={
            "suggestion": "Simple suggestion",
            "confidence": 0.7
        })
        
        suggestion_request = {
            "message": "Simple request",
            "include_sources": False,
            "use_knowledge_graph": False
        }
        
        response = client.post("/api/suggestion", json=suggestion_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["suggestion"] == "Simple suggestion"
        assert data["sources"] is None
        assert data["entities_used"] is None


class TestKnowledgeGraphEndpoints:
    """Test knowledge graph endpoints"""
    
    @patch('api_server.knowledge_graph')
    def test_knowledge_graph_find_related(self, mock_kg):
        """Test finding related entities"""
        mock_entity = Mock()
        mock_entity.text = "related entity"
        mock_entity.entity_type = "CONCEPT"
        mock_entity.frequency = 5
        
        mock_kg.find_related_entities = AsyncMock(return_value=[
            (mock_entity, 0.9)
        ])
        
        request_data = {
            "operation": "find_related",
            "entity": "test entity",
            "max_hops": 2,
            "limit": 10
        }
        
        response = client.post("/api/knowledge-graph", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["entity"] == "test entity"
        assert len(data["related_entities"]) == 1
        assert data["related_entities"][0]["text"] == "related entity"
    
    @patch('api_server.knowledge_graph')
    def test_knowledge_graph_get_clusters(self, mock_kg):
        """Test getting concept clusters"""
        mock_cluster = Mock()
        mock_cluster.cluster_id = "cluster1"
        mock_cluster.name = "Test Cluster"
        mock_cluster.entity_ids = {"entity1", "entity2"}
        mock_cluster.keywords = ["test", "cluster"]
        mock_cluster.coherence_score = 0.8
        
        mock_kg.find_concept_clusters = AsyncMock(return_value=[mock_cluster])
        
        request_data = {
            "operation": "get_clusters",
            "entities": ["entity1", "entity2"]
        }
        
        response = client.post("/api/knowledge-graph", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["clusters"]) == 1
        assert data["clusters"][0]["name"] == "Test Cluster"
    
    @patch('api_server.knowledge_graph')
    def test_knowledge_graph_get_stats(self, mock_kg):
        """Test getting knowledge graph statistics"""
        mock_stats = Mock()
        mock_stats.total_entities = 100
        mock_stats.total_relationships = 250
        mock_stats.total_clusters = 10
        mock_stats.entity_types = {"PERSON": 30, "CONCEPT": 70}
        mock_stats.relationship_types = {"RELATED_TO": 150, "MENTIONS": 100}
        mock_stats.avg_relationships_per_entity = 2.5
        mock_stats.graph_density = 0.05
        mock_stats.most_connected_entities = ["entity1", "entity2"]
        mock_stats.largest_clusters = ["cluster1", "cluster2"]
        
        mock_kg.get_statistics = Mock(return_value=mock_stats)
        
        request_data = {"operation": "get_stats"}
        
        response = client.post("/api/knowledge-graph", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_entities"] == 100
        assert data["total_relationships"] == 250
        assert data["graph_density"] == 0.05
    
    def test_knowledge_graph_invalid_operation(self):
        """Test invalid knowledge graph operation"""
        request_data = {"operation": "invalid_operation"}
        
        response = client.post("/api/knowledge-graph", json=request_data)
        
        assert response.status_code == 400
        assert "Unknown operation" in response.json()["detail"]
    
    @patch('api_server.knowledge_graph')
    def test_build_knowledge_clusters(self, mock_kg):
        """Test building knowledge clusters"""
        mock_stats = Mock()
        mock_stats.total_clusters = 5
        
        mock_kg.build_concept_clusters = AsyncMock()
        mock_kg.get_statistics = Mock(return_value=mock_stats)
        
        response = client.post("/api/knowledge-graph/build-clusters")
        
        assert response.status_code == 200
        data = response.json()
        assert "Clusters built successfully" in data["message"]
        assert data["total_clusters"] == 5


class TestCollectionEndpoints:
    """Test collection management endpoints"""
    
    @patch('api_server.search_system')
    def test_list_collections(self, mock_search):
        """Test listing all collections"""
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_collection.count = Mock(return_value=50)
        
        mock_search.client.list_collections = Mock(return_value=[mock_collection])
        
        response = client.get("/api/collections")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "test_collection"
        assert data[0]["document_count"] == 50
    
    @patch('api_server.search_system')
    def test_get_collection(self, mock_search):
        """Test getting specific collection"""
        mock_collection = Mock()
        mock_collection.name = "specific_collection"
        mock_collection.count = Mock(return_value=25)
        
        mock_search.client.get_collection = Mock(return_value=mock_collection)
        
        response = client.get("/api/collections/specific_collection")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "specific_collection"
        assert data["document_count"] == 25
    
    @patch('api_server.search_system')
    def test_get_collection_not_found(self, mock_search):
        """Test getting non-existent collection"""
        mock_search.client.get_collection = Mock(return_value=None)
        
        response = client.get("/api/collections/nonexistent")
        
        assert response.status_code == 404
        assert "Collection not found" in response.json()["detail"]


class TestSystemEndpoints:
    """Test system management endpoints"""
    
    @patch('api_server.memory_manager')
    @patch('api_server.search_system')
    @patch('api_server.provider_manager')
    @patch('api_server.concurrent_optimizer')
    @patch('api_server.monitoring')
    def test_health_check(self, mock_monitoring, mock_optimizer, mock_provider, mock_search, mock_memory):
        """Test system health check"""
        # Mock components
        mock_memory.get_memory_status = Mock(return_value={"used": 50.0, "available": 50.0})
        
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_collection.count = Mock(return_value=10)
        mock_search.client.list_collections = Mock(return_value=[mock_collection])
        
        mock_provider.list_providers = Mock(return_value=[
            {"name": "openai", "active": True}
        ])
        
        mock_optimizer.get_active_operations = Mock(return_value=["op1", "op2"])
        mock_monitoring.get_uptime = Mock(return_value=3600.0)
        
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["uptime"] == 3600.0
        assert data["active_operations"] == 2
        assert "memory_usage" in data
        assert "collection_stats" in data
        assert "provider_status" in data
    
    @patch('api_server.memory_manager')
    @patch('api_server.concurrent_optimizer')
    def test_optimize_system(self, mock_optimizer, mock_memory):
        """Test system optimization"""
        mock_memory.optimize_memory = Mock()
        mock_optimizer.optimize_system = AsyncMock(return_value={"optimized": True})
        
        response = client.post("/api/system/optimize")
        
        assert response.status_code == 200
        data = response.json()
        assert "System optimization completed" in data["message"]
        assert "results" in data
    
    @patch('api_server.monitoring')
    def test_get_system_metrics(self, mock_monitoring):
        """Test getting system metrics"""
        mock_monitoring.get_metrics = AsyncMock(return_value={
            "performance": {"avg_response_time": 0.5},
            "usage": {"total_requests": 100}
        })
        
        response = client.get("/api/system/metrics?metric_type=performance&time_range=1h")
        
        assert response.status_code == 200
        data = response.json()
        assert "performance" in data
        assert "usage" in data


class TestProviderEndpoints:
    """Test LLM provider endpoints"""
    
    @patch('api_server.provider_manager')
    def test_list_providers(self, mock_provider):
        """Test listing providers"""
        mock_provider.list_providers = Mock(return_value=[
            {"name": "openai", "model": "gpt-4", "active": True},
            {"name": "anthropic", "model": "claude-3", "active": False}
        ])
        
        response = client.get("/api/providers")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["providers"]) == 2
        assert data["providers"][0]["name"] == "openai"
    
    @patch('api_server.provider_manager')
    def test_configure_provider_legacy(self, mock_provider):
        """Test configuring provider (legacy endpoint)"""
        mock_provider.add_provider = Mock(return_value={"success": True})
        
        config_data = {
            "provider": "openai",
            "api_key": "test-key",
            "model": "gpt-4"
        }
        
        response = client.post("/api/llm/providers/configure", json=config_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "configured successfully" in data["message"]
    
    @patch('api_server.provider_manager')
    def test_activate_provider_legacy(self, mock_provider):
        """Test activating provider (legacy endpoint)"""
        mock_provider.set_active_provider = Mock(return_value={"success": True})
        
        request_data = {"provider": "openai"}
        
        response = client.post("/api/llm/providers/activate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "activated successfully" in data["message"]
    
    @patch('api_server.provider_manager')
    def test_test_provider_legacy(self, mock_provider):
        """Test testing provider (legacy endpoint)"""
        mock_provider.get_active_provider = Mock(return_value={"name": "openai"})
        mock_provider.generate = AsyncMock(return_value={"content": "test successful"})
        
        response = client.post("/api/llm/providers/test")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "test successful" in data["response"]
    
    @patch('api_server.provider_manager')
    def test_activate_provider_new(self, mock_provider):
        """Test activating provider (new endpoint)"""
        mock_provider.set_active_provider = Mock(return_value={"success": True})
        
        response = client.post("/api/providers/openai/activate")
        
        assert response.status_code == 200
        data = response.json()
        assert "activated successfully" in data["message"]


class TestLegacyEndpoints:
    """Test legacy compatibility endpoints"""
    
    @patch('api_server.provider_manager')
    def test_get_current_llm_legacy(self, mock_provider):
        """Test getting current LLM (legacy endpoint)"""
        mock_provider.list_providers = Mock(return_value=[
            {"name": "openai", "model": "gpt-4", "active": True}
        ])
        
        response = client.get("/api/llm/current")
        
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4"
        assert data["status"] == "active"
    
    def test_create_conversation_legacy(self):
        """Test creating conversation (legacy endpoint)"""
        request_data = {"conversation_id": "test-channel"}
        
        response = client.post("/api/conversations", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "test-channel"
        assert data["status"] == "created"
    
    @patch('api_server.search_system')
    def test_check_messages_legacy(self, mock_search):
        """Test checking messages (legacy endpoint)"""
        mock_search.search_multiple_collections = AsyncMock(return_value=[
            {"id": "msg1", "content": "test message"}
        ])
        
        response = client.get("/api/conversations/test-channel/has-messages")
        
        assert response.status_code == 200
        data = response.json()
        assert data["has_messages"] is True
        assert data["count"] == 1
    
    @patch('api_server.search_system')
    @patch('api_server.knowledge_graph')
    def test_store_messages_bulk_legacy(self, mock_kg, mock_search):
        """Test bulk message storage (legacy endpoint)"""
        mock_search.store_content = AsyncMock()
        mock_entity = Mock()
        mock_entity.text = "test entity"
        mock_entity.entity_type = "CONCEPT"
        mock_kg.extract_entities = Mock(return_value=[mock_entity])
        
        messages = [
            {
                "content": "Test message 1",
                "user_id": "user1",
                "channel_id": "channel1",
                "timestamp": "123456789"
            },
            {
                "content": "Test message 2", 
                "user_id": "user2",
                "channel_id": "channel1",
                "timestamp": "123456790"
            }
        ]
        
        response = client.post("/api/messages/bulk", json=messages)
        
        assert response.status_code == 200
        data = response.json()
        assert data["stored"] == 2
        assert data["total"] == 2
        assert data["status"] == "completed"


class TestRootEndpoint:
    """Test root and error handling"""
    
    def test_root_endpoint(self):
        """Test API root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Docling Knowledge API"
        assert data["version"] == "3.0.0"
        assert data["status"] == "operational"
        assert "/docs" in data["documentation"]


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_missing_required_fields(self):
        """Test missing required fields in requests"""
        # Test search without query
        response = client.post("/api/search", json={})
        assert response.status_code == 422  # Validation error
        
        # Test knowledge graph without operation
        response = client.post("/api/knowledge-graph", json={})
        assert response.status_code == 422  # Validation error
    
    def test_invalid_document_id(self):
        """Test invalid document ID handling"""
        # Test with special characters that might cause issues
        response = client.get("/api/documents/invalid/id/with/slashes")
        assert response.status_code in [404, 422]  # Not found or validation error
    
    @patch('api_server.search_system')
    def test_search_system_error(self, mock_search):
        """Test handling of search system errors"""
        mock_search.search_multiple_collections = AsyncMock(side_effect=Exception("Search failed"))
        
        search_request = {"query": "test query"}
        response = client.post("/api/search", json=search_request)
        
        assert response.status_code == 500
        assert "Search failed" in response.json()["detail"]


# Test runner
if __name__ == "__main__":
    # Run specific test classes or all tests
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        pytest.main([f"-v", f"test_api_endpoints.py::{test_class}"])
    else:
        pytest.main(["-v", "test_api_endpoints.py"])