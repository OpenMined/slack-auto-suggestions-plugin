"""
Vector Database Manager for Slack Messages using ChromaDB
"""
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Any
import hashlib
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDBManager:
    """Manages vector storage and retrieval for Slack messages using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize ChromaDB client and collections
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Name of the sentence-transformer model to use
        """
        logger.info(f"Initializing VectorDBManager with persist_directory: {persist_directory}")
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use sentence-transformers for local embeddings (privacy-friendly)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Create collections for different data types
        self.messages_collection = self._get_or_create_collection("slack_messages")
        self.documents_collection = self._get_or_create_collection("slack_documents")
        self.threads_collection = self._get_or_create_collection("slack_threads")
        
        logger.info("VectorDBManager initialized successfully")
    
    def _get_or_create_collection(self, name: str):
        """Get or create a collection with the embedding function"""
        try:
            # When getting an existing collection, we need to pass the embedding function
            collection = self.client.get_collection(
                name=name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Retrieved existing collection: {name}")
        except:
            collection = self.client.create_collection(
                name=name,
                embedding_function=self.embedding_function,
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "hnsw:space": "cosine"  # Use cosine similarity instead of L2
                }
            )
            logger.info(f"Created new collection: {name}")
        
        return collection
    
    def _generate_message_id(self, message: Dict) -> str:
        """Generate unique ID for a message based on conversation_id and timestamp"""
        content = f"{message['conversation_id']}_{message['ts']}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _prepare_message_for_embedding(self, message: Dict) -> str:
        """Prepare message text for embedding with context"""
        # Include user name and message text for better context
        text_parts = [f"User: {message['user_name']}"]
        text_parts.append(f"Message: {message['text']}")
        
        # Add thread context if available
        if message.get('thread_ts') and message['thread_ts'] != message['ts']:
            text_parts.append("(in thread)")
        
        return " | ".join(text_parts)
    
    def add_message(self, message: Dict) -> str:
        """
        Add a single message to the vector database
        
        Args:
            message: Dictionary containing message data
            
        Returns:
            Generated message ID
        """
        msg_id = self._generate_message_id(message)
        
        # Check if message already exists
        existing = self.messages_collection.get(ids=[msg_id])
        if existing['ids']:
            logger.debug(f"Message {msg_id} already exists, skipping")
            return msg_id
        
        # Prepare document text for embedding
        document_text = self._prepare_message_for_embedding(message)
        
        # Prepare metadata (ChromaDB has limits on metadata) - ensure no None values
        metadata = {
            "conversation_id": str(message.get("conversation_id", "")),
            "user_id": str(message.get("user_id", "")),
            "user_name": str(message.get("user_name", "")),
            "timestamp": str(message.get("ts", "")),
            "has_attachments": str(message.get("has_attachments", False)),
            "created_at": str(message.get("created_at", datetime.now().isoformat())),
            "thread_ts": str(message.get("thread_ts", ""))
        }
        
        # Add to ChromaDB
        self.messages_collection.add(
            documents=[document_text],
            metadatas=[metadata],
            ids=[msg_id]
        )
        
        logger.debug(f"Added message {msg_id} to vector database")
        return msg_id
    
    def add_messages_batch(self, messages: List[Dict]) -> List[str]:
        """
        Add multiple messages in batch for efficiency
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of generated message IDs
        """
        if not messages:
            return []
        
        # Prepare batch data
        ids = []
        documents = []
        metadatas = []
        new_messages = []
        
        # Generate IDs and check for existing messages
        all_ids = [self._generate_message_id(msg) for msg in messages]
        existing = self.messages_collection.get(ids=all_ids)
        existing_ids = set(existing['ids'])
        
        for message in messages:
            msg_id = self._generate_message_id(message)
            
            # Skip if already exists
            if msg_id in existing_ids:
                logger.debug(f"Message {msg_id} already exists, skipping")
                continue
            
            ids.append(msg_id)
            new_messages.append(message)
            
            # Prepare document text
            document_text = self._prepare_message_for_embedding(message)
            documents.append(document_text)
            
            # Prepare metadata - ensure no None values
            metadata = {
                "conversation_id": str(message.get("conversation_id", "")),
                "user_id": str(message.get("user_id", "")),
                "user_name": str(message.get("user_name", "")),
                "timestamp": str(message.get("ts", "")),
                "has_attachments": str(message.get("has_attachments", False)),
                "created_at": str(message.get("created_at", datetime.now().isoformat())),
                "thread_ts": str(message.get("thread_ts", ""))
            }
            metadatas.append(metadata)
        
        # Batch add to ChromaDB if there are new messages
        if ids:
            self.messages_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(ids)} new messages to vector database")
        
        return ids
    
    def search_messages(self, 
                       query: str, 
                       n_results: int = 10,
                       conversation_id: Optional[str] = None,
                       user_id: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> List[Dict]:
        """
        Search messages using semantic search with optional filters
        
        Args:
            query: Search query text
            n_results: Number of results to return
            conversation_id: Filter by conversation
            user_id: Filter by user
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            
        Returns:
            List of formatted search results
        """
        # Build where clause for metadata filtering
        where_clause = {}
        
        if conversation_id:
            where_clause["conversation_id"] = conversation_id
        if user_id:
            where_clause["user_id"] = user_id
        
        # Date filtering would require more complex logic
        # ChromaDB doesn't support range queries directly
        
        # Perform semantic search
        logger.info(f"Searching with query: '{query}', n_results: {n_results}, where: {where_clause}")
        
        try:
            results = self.messages_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            logger.info(f"Query returned {len(results.get('ids', [[]])[0])} results")
            logger.debug(f"Raw results: {results}")
            
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []
        
        return self._format_search_results(results)
    
    def search_by_thread(self, thread_ts: str, n_results: int = 50) -> List[Dict]:
        """
        Get all messages in a specific thread
        
        Args:
            thread_ts: Thread timestamp
            n_results: Maximum number of results
            
        Returns:
            List of messages in the thread
        """
        results = self.messages_collection.query(
            query_texts=[""],  # Empty query for metadata-only search
            n_results=n_results,
            where={"thread_ts": thread_ts},
            include=["documents", "metadatas"]
        )
        
        return self._format_search_results(results)
    
    def get_similar_conversations(self, 
                                 message_text: str, 
                                 n_results: int = 5,
                                 exclude_conversation_id: Optional[str] = None) -> List[Dict]:
        """
        Find conversations with similar content
        
        Args:
            message_text: Reference message text
            n_results: Number of similar conversations to find
            exclude_conversation_id: Conversation to exclude from results
            
        Returns:
            List of similar conversations with sample messages
        """
        # Search for similar messages
        where_clause = {}
        if exclude_conversation_id:
            where_clause = {"conversation_id": {"$ne": exclude_conversation_id}}
        
        results = self.messages_collection.query(
            query_texts=[message_text],
            n_results=n_results * 3,  # Get more results to ensure variety
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Group by conversation and return top conversations
        conversations = {}
        for result in self._format_search_results(results):
            conv_id = result['metadata']['conversation_id']
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(result)
        
        # Sort conversations by best match distance
        sorted_convs = sorted(
            conversations.items(), 
            key=lambda x: min(msg['distance'] for msg in x[1])
        )
        
        return [
            {
                "conversation_id": conv_id,
                "sample_messages": messages[:3],
                "best_match_distance": min(msg['distance'] for msg in messages)
            }
            for conv_id, messages in sorted_convs[:n_results]
        ]
    
    def _format_search_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results into a more usable format"""
        formatted_results = []
        
        if not results['ids'] or not results['ids'][0]:
            return formatted_results
        
        for i, doc_id in enumerate(results['ids'][0]):
            result = {
                'id': doc_id,
                'text': results['documents'][0][i] if 'documents' in results else None,
                'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
            }
            
            # Add distance if available (lower is more similar)
            if 'distances' in results and results['distances']:
                result['distance'] = results['distances'][0][i]
                # For cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity score: 1 = identical, -1 = opposite
                result['similarity_score'] = 1 - result['distance']  # This gives range [1, -1]
            
            formatted_results.append(result)
        
        return formatted_results
    
    def add_document(self, content: str, metadata: Dict) -> str:
        """
        Add a document chunk to the vector database
        
        Args:
            content: Document content text
            metadata: Document metadata dictionary
            
        Returns:
            Generated document chunk ID
        """
        # Generate unique ID for this document chunk
        doc_id = metadata.get('id') or self._generate_document_id(
            metadata.get('filename', 'unknown'), 
            metadata.get('chunk_index', 0)
        )
        
        # Check if document chunk already exists
        try:
            existing = self.documents_collection.get(ids=[doc_id])
            if existing['ids']:
                logger.debug(f"Document chunk {doc_id} already exists, skipping")
                return doc_id
        except Exception as e:
            logger.warning(f"Error checking for existing document chunk: {e}")
        
        # Ensure all metadata values are strings (ChromaDB requirement)
        clean_metadata = {}
        for key, value in metadata.items():
            if key != 'id':  # Don't include id in metadata
                clean_metadata[key] = str(value) if value is not None else ""
        
        try:
            # Add to ChromaDB documents collection
            self.documents_collection.add(
                documents=[content],
                metadatas=[clean_metadata],
                ids=[doc_id]
            )
            
            logger.debug(f"Added document chunk {doc_id} to vector database")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document chunk to vector database: {e}")
            raise
    
    def _generate_document_id(self, filename: str, chunk_index: int = 0) -> str:
        """Generate unique ID for a document chunk"""
        content = f"doc_{filename}_{chunk_index}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def search_documents(self, 
                        query: str, 
                        n_results: int = 10,
                        filename: Optional[str] = None,
                        document_type: Optional[str] = None) -> List[Dict]:
        """
        Search documents using semantic search with optional filters
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filename: Filter by specific filename
            document_type: Filter by document type
            
        Returns:
            List of formatted search results
        """
        # Build where clause for metadata filtering
        where_clause = {"document_type": "knowledge_base"}
        
        if filename:
            where_clause["filename"] = filename
        if document_type:
            where_clause["file_type"] = document_type
        
        # Perform semantic search
        logger.info(f"Searching documents with query: '{query}', n_results: {n_results}, where: {where_clause}")
        
        try:
            results = self.documents_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            num_results = len(results.get('ids', [[]])[0])
            logger.info(f"Document query returned {num_results} results")
            
            # Log details of found documents for debugging
            if num_results > 0 and results.get('metadatas') and results.get('distances'):
                logger.debug("=== DOCUMENT SEARCH RESULTS ===")
                for i in range(num_results):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    similarity = 1 - distance
                    filename = metadata.get('filename', 'Unknown')
                    chunk_idx = metadata.get('chunk_index', '0')
                    logger.debug(f"Result {i+1}: {filename} (chunk {chunk_idx}) - Similarity: {similarity:.3f}")
                logger.debug("=== END DOCUMENT SEARCH RESULTS ===")
            
        except Exception as e:
            logger.error(f"Error during document search: {e}")
            return []
        
        return self._format_search_results(results)
    
    def delete_documents_by_filename(self, filename: str) -> int:
        """
        Delete all document chunks for a specific filename
        
        Args:
            filename: Filename to delete
            
        Returns:
            Number of deleted chunks
        """
        try:
            # Get all chunks for this file
            results = self.documents_collection.get(
                where={"filename": filename}
            )
            
            if results['ids']:
                # Delete all found chunks
                self.documents_collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} document chunks for filename {filename}")
                return len(results['ids'])
            
            return 0
            
        except Exception as e:
            logger.error(f"Error deleting documents for filename {filename}: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database collections"""
        stats = {
            "messages": {
                "count": self.messages_collection.count(),
                "name": "slack_messages"
            },
            "documents": {
                "count": self.documents_collection.count(),
                "name": "slack_documents"
            },
            "threads": {
                "count": self.threads_collection.count(),
                "name": "slack_threads"
            },
            "total_items": 0
        }
        
        stats["total_items"] = sum(col["count"] for col in stats.values() if isinstance(col, dict))
        
        return stats
    
    def delete_conversation_vectors(self, conversation_id: str) -> int:
        """
        Delete all vectors for a specific conversation
        
        Args:
            conversation_id: Conversation ID to delete
            
        Returns:
            Number of deleted items
        """
        # Get all messages for this conversation
        results = self.messages_collection.get(
            where={"conversation_id": conversation_id}
        )
        
        if results['ids']:
            # Delete all found messages
            self.messages_collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} vectors for conversation {conversation_id}")
            return len(results['ids'])
        
        return 0
    
    def reset_collection(self, collection_name: str) -> bool:
        """
        Reset (delete and recreate) a collection
        
        Args:
            collection_name: Name of collection to reset
            
        Returns:
            Success status
        """
        try:
            self.client.delete_collection(name=collection_name)
            
            # Recreate based on collection type
            if collection_name == "slack_messages":
                self.messages_collection = self._get_or_create_collection(collection_name)
            elif collection_name == "slack_documents":
                self.documents_collection = self._get_or_create_collection(collection_name)
            elif collection_name == "slack_threads":
                self.threads_collection = self._get_or_create_collection(collection_name)
            
            logger.info(f"Reset collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection {collection_name}: {e}")
            return False