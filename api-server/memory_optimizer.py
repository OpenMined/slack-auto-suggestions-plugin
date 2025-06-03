"""
Phase 2: Memory Management Optimization

Advanced memory management system for handling large documents, complex hierarchies,
and optimizing memory usage across multi-collection operations.
"""

import asyncio
import logging
import gc
import psutil
import weakref
from typing import Dict, List, Optional, Any, Tuple, Set, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import sys
import tracemalloc
from collections import defaultdict, OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import mmap
import tempfile
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)


class MemoryPressureLevel(Enum):
    """Memory pressure levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class CacheEvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


class MemoryPoolType(Enum):
    """Types of memory pools"""
    DOCUMENT_PROCESSING = "document_processing"
    SEARCH_RESULTS = "search_results"
    EMBEDDINGS = "embeddings"
    TEMPORARY_DATA = "temporary_data"
    CACHE = "cache"


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    timestamp: datetime
    
    # System memory
    total_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    memory_percent: float = 0.0
    
    # Process memory
    process_memory_mb: float = 0.0
    process_memory_percent: float = 0.0
    
    # Memory pools
    pool_usage: Dict[str, float] = field(default_factory=dict)
    
    # Cache memory
    cache_memory_mb: float = 0.0
    
    # Peak memory usage
    peak_memory_mb: float = 0.0
    
    # Memory pressure level
    pressure_level: MemoryPressureLevel = MemoryPressureLevel.LOW


@dataclass
class MemoryPool:
    """Memory pool for specific data types"""
    name: str
    pool_type: MemoryPoolType
    max_size_mb: float
    current_size_mb: float = 0.0
    
    # Pool configuration
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU
    enable_compression: bool = False
    enable_disk_overflow: bool = False
    
    # Usage tracking
    access_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    
    # Data storage
    data: OrderedDict = field(default_factory=OrderedDict)
    access_times: Dict[str, datetime] = field(default_factory=dict)
    access_frequency: Dict[str, int] = field(default_factory=dict)
    
    # Disk overflow
    overflow_directory: Optional[Path] = None


@dataclass
class LargeDocumentHandle:
    """Handle for managing large documents in memory"""
    document_id: str
    total_size_mb: float
    
    # Component references
    sections: List[weakref.ref] = field(default_factory=list)
    chunks: List[weakref.ref] = field(default_factory=list)
    tables: List[weakref.ref] = field(default_factory=list)
    entities: List[weakref.ref] = field(default_factory=list)
    
    # Memory management
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    is_cached: bool = False
    cache_priority: float = 0.0
    
    # Streaming support
    is_streaming: bool = False
    loaded_components: Set[str] = field(default_factory=set)


class MemoryManager:
    """
    Advanced memory management system for multi-collection operations
    """
    
    def __init__(self):
        # Memory configuration
        self.memory_config = {
            # Memory limits (MB)
            "max_total_memory_mb": 4096,  # 4GB
            "warning_threshold_percent": 80,
            "critical_threshold_percent": 95,
            
            # Pool configurations
            "document_processing_pool_mb": 1024,  # 1GB
            "search_results_pool_mb": 512,  # 512MB
            "embeddings_pool_mb": 1024,  # 1GB
            "cache_pool_mb": 1024,  # 1GB
            "temporary_data_pool_mb": 512,  # 512MB
            
            # Large document handling
            "large_document_threshold_mb": 50,  # 50MB
            "enable_document_streaming": True,
            "max_concurrent_large_docs": 5,
            
            # Garbage collection
            "enable_aggressive_gc": True,
            "gc_frequency_seconds": 60,
            
            # Compression
            "enable_compression": True,
            "compression_threshold_mb": 10,
            
            # Disk overflow
            "enable_disk_overflow": True,
            "overflow_directory": "memory_overflow"
        }
        
        # Memory pools
        self.memory_pools: Dict[str, MemoryPool] = {}
        
        # Large document tracking
        self.large_documents: Dict[str, LargeDocumentHandle] = {}
        self.active_large_docs: Set[str] = set()
        
        # Memory monitoring
        self.memory_stats_history: List[MemoryStats] = []
        self.memory_pressure_callbacks: List[callable] = []
        
        # Threading
        self.memory_monitor_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory_mgr")
        
        # Performance tracking
        self.performance_stats = {
            "memory_optimizations": 0,
            "documents_streamed": 0,
            "cache_evictions": 0,
            "gc_collections": 0,
            "overflow_operations": 0
        }
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize the memory manager"""
        try:
            logger.info("Initializing Memory Manager")
            
            # Enable memory tracing if available
            if not tracemalloc.is_tracing():
                tracemalloc.start(10)  # Keep 10 frames
            
            # Create memory pools
            await self._create_memory_pools()
            
            # Setup disk overflow directory
            if self.memory_config["enable_disk_overflow"]:
                await self._setup_disk_overflow()
            
            # Start memory monitoring
            asyncio.create_task(self._memory_monitoring_loop())
            
            # Start garbage collection task
            if self.memory_config["enable_aggressive_gc"]:
                asyncio.create_task(self._garbage_collection_loop())
            
            self.initialized = True
            logger.info("Memory Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory Manager: {e}")
            raise
    
    async def _create_memory_pools(self):
        """Create and configure memory pools"""
        pool_configs = [
            (MemoryPoolType.DOCUMENT_PROCESSING, self.memory_config["document_processing_pool_mb"]),
            (MemoryPoolType.SEARCH_RESULTS, self.memory_config["search_results_pool_mb"]),
            (MemoryPoolType.EMBEDDINGS, self.memory_config["embeddings_pool_mb"]),
            (MemoryPoolType.CACHE, self.memory_config["cache_pool_mb"]),
            (MemoryPoolType.TEMPORARY_DATA, self.memory_config["temporary_data_pool_mb"])
        ]
        
        for pool_type, max_size_mb in pool_configs:
            pool = MemoryPool(
                name=pool_type.value,
                pool_type=pool_type,
                max_size_mb=max_size_mb,
                eviction_policy=CacheEvictionPolicy.ADAPTIVE,
                enable_compression=self.memory_config["enable_compression"],
                enable_disk_overflow=self.memory_config["enable_disk_overflow"]
            )
            
            if pool.enable_disk_overflow:
                overflow_dir = Path(self.memory_config["overflow_directory"]) / pool.name
                overflow_dir.mkdir(parents=True, exist_ok=True)
                pool.overflow_directory = overflow_dir
            
            self.memory_pools[pool.name] = pool
        
        logger.info(f"Created {len(self.memory_pools)} memory pools")
    
    async def _setup_disk_overflow(self):
        """Setup disk overflow directory"""
        overflow_dir = Path(self.memory_config["overflow_directory"])
        overflow_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up old overflow files
        for file_path in overflow_dir.glob("*.overflow"):
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean overflow file {file_path}: {e}")
    
    async def allocate_document_memory(
        self,
        document_id: str,
        estimated_size_mb: float,
        components: Dict[str, List[Any]]
    ) -> LargeDocumentHandle:
        """
        Allocate memory for document processing with optimization for large documents
        """
        if not self.initialized:
            await self.initialize()
        
        logger.debug(f"Allocating memory for document {document_id}: {estimated_size_mb:.2f}MB")
        
        try:
            # Check if this is a large document
            is_large = estimated_size_mb > self.memory_config["large_document_threshold_mb"]
            
            # Create document handle
            handle = LargeDocumentHandle(
                document_id=document_id,
                total_size_mb=estimated_size_mb,
                is_streaming=is_large and self.memory_config["enable_document_streaming"]
            )
            
            # Handle large document with streaming
            if handle.is_streaming:
                await self._setup_document_streaming(handle, components)
            else:
                await self._load_document_fully(handle, components)
            
            # Register document
            self.large_documents[document_id] = handle
            
            if is_large:
                await self._manage_large_document_concurrency(document_id)
            
            return handle
            
        except Exception as e:
            logger.error(f"Failed to allocate memory for document {document_id}: {e}")
            raise
    
    async def _setup_document_streaming(
        self,
        handle: LargeDocumentHandle,
        components: Dict[str, List[Any]]
    ):
        """Setup streaming for large documents"""
        logger.info(f"Setting up streaming for large document {handle.document_id}")
        
        # Store component references with weak references
        for component_type, component_list in components.items():
            component_refs = []
            
            for component in component_list:
                # Create weak reference to component
                ref = weakref.ref(component, self._component_cleanup_callback)
                component_refs.append(ref)
            
            # Store references in handle
            if component_type == "sections":
                handle.sections = component_refs
            elif component_type == "chunks":
                handle.chunks = component_refs
            elif component_type == "tables":
                handle.tables = component_refs
            elif component_type == "entities":
                handle.entities = component_refs
        
        # Initially load only essential components
        await self._load_essential_components(handle)
    
    async def _load_document_fully(
        self,
        handle: LargeDocumentHandle,
        components: Dict[str, List[Any]]
    ):
        """Load document fully into memory"""
        pool = self.memory_pools[MemoryPoolType.DOCUMENT_PROCESSING.value]
        
        # Check if we have enough memory
        available_memory = pool.max_size_mb - pool.current_size_mb
        
        if handle.total_size_mb > available_memory:
            # Need to free up memory
            await self._free_pool_memory(pool, handle.total_size_mb)
        
        # Store components in memory pool
        for component_type, component_list in components.items():
            pool_key = f"{handle.document_id}_{component_type}"
            await self._store_in_pool(pool, pool_key, component_list)
        
        pool.current_size_mb += handle.total_size_mb
        handle.is_cached = True
    
    async def _load_essential_components(self, handle: LargeDocumentHandle):
        """Load only essential components for streaming documents"""
        # Load document metadata and structure
        essential_components = ["sections"]  # Start with sections for navigation
        
        for component_type in essential_components:
            if component_type not in handle.loaded_components:
                await self._load_component_type(handle, component_type)
                handle.loaded_components.add(component_type)
    
    async def _load_component_type(self, handle: LargeDocumentHandle, component_type: str):
        """Load a specific component type for a streaming document"""
        logger.debug(f"Loading {component_type} for document {handle.document_id}")
        
        # This would load the specific component type from disk or recreate from source
        # For now, mark as loaded
        handle.access_count += 1
        handle.last_accessed = datetime.now()
    
    async def _manage_large_document_concurrency(self, document_id: str):
        """Manage concurrency for large documents"""
        self.active_large_docs.add(document_id)
        
        # Check concurrency limit
        max_concurrent = self.memory_config["max_concurrent_large_docs"]
        
        if len(self.active_large_docs) > max_concurrent:
            # Need to evict least recently used large document
            await self._evict_lru_large_document()
    
    async def _evict_lru_large_document(self):
        """Evict least recently used large document"""
        if not self.active_large_docs:
            return
        
        # Find LRU document
        lru_doc_id = None
        lru_time = datetime.now()
        
        for doc_id in self.active_large_docs:
            if doc_id in self.large_documents:
                handle = self.large_documents[doc_id]
                if handle.last_accessed < lru_time:
                    lru_time = handle.last_accessed
                    lru_doc_id = doc_id
        
        if lru_doc_id:
            await self._evict_large_document(lru_doc_id)
    
    async def _evict_large_document(self, document_id: str):
        """Evict a large document from memory"""
        logger.debug(f"Evicting large document {document_id}")
        
        if document_id in self.large_documents:
            handle = self.large_documents[document_id]
            
            # Clear loaded components
            handle.loaded_components.clear()
            handle.is_cached = False
            
            # Remove from active set
            self.active_large_docs.discard(document_id)
            
            # Update performance stats
            self.performance_stats["cache_evictions"] += 1
    
    def _component_cleanup_callback(self, weak_ref):
        """Callback for when a component is garbage collected"""
        # Component has been garbage collected
        pass
    
    async def get_memory_pool(self, pool_type: MemoryPoolType) -> MemoryPool:
        """Get a specific memory pool"""
        return self.memory_pools.get(pool_type.value)
    
    async def _store_in_pool(self, pool: MemoryPool, key: str, data: Any):
        """Store data in a memory pool with eviction handling"""
        
        # Estimate data size
        data_size_mb = await self._estimate_data_size(data)
        
        # Check if we need to free space
        if pool.current_size_mb + data_size_mb > pool.max_size_mb:
            await self._free_pool_memory(pool, data_size_mb)
        
        # Compress data if enabled and above threshold
        if (pool.enable_compression and 
            data_size_mb > self.memory_config["compression_threshold_mb"]):
            data = await self._compress_data(data)
        
        # Store data
        pool.data[key] = data
        pool.access_times[key] = datetime.now()
        pool.access_frequency[key] = pool.access_frequency.get(key, 0) + 1
        pool.current_size_mb += data_size_mb
        pool.access_count += 1
    
    async def _free_pool_memory(self, pool: MemoryPool, required_mb: float):
        """Free memory in a pool using the configured eviction policy"""
        logger.debug(f"Freeing {required_mb:.2f}MB from pool {pool.name}")
        
        freed_mb = 0.0
        
        if pool.eviction_policy == CacheEvictionPolicy.LRU:
            # Sort by access time (oldest first)
            items_by_access = sorted(
                pool.access_times.items(),
                key=lambda x: x[1]
            )
        elif pool.eviction_policy == CacheEvictionPolicy.LFU:
            # Sort by frequency (least frequent first)
            items_by_frequency = sorted(
                pool.access_frequency.items(),
                key=lambda x: x[1]
            )
            items_by_access = items_by_frequency
        elif pool.eviction_policy == CacheEvictionPolicy.FIFO:
            # Use insertion order
            items_by_access = list(pool.data.items())
        else:  # ADAPTIVE
            # Use a combination of recency and frequency
            items_by_score = []
            current_time = datetime.now()
            
            for key in pool.data:
                last_access = pool.access_times.get(key, current_time)
                frequency = pool.access_frequency.get(key, 1)
                
                # Calculate adaptive score (lower is better for eviction)
                time_since_access = (current_time - last_access).total_seconds()
                score = time_since_access / max(frequency, 1)
                
                items_by_score.append((key, score))
            
            items_by_access = sorted(items_by_score, key=lambda x: x[1], reverse=True)
        
        # Evict items until we have enough space
        for key, _ in items_by_access:
            if freed_mb >= required_mb:
                break
            
            if key in pool.data:
                # Move to disk overflow if enabled
                if pool.enable_disk_overflow:
                    await self._move_to_disk_overflow(pool, key)
                else:
                    # Remove from memory
                    del pool.data[key]
                
                # Clean up tracking data
                pool.access_times.pop(key, None)
                pool.access_frequency.pop(key, None)
                
                # Estimate freed memory
                freed_mb += 10  # Simplified estimation
                pool.current_size_mb = max(0, pool.current_size_mb - 10)
                pool.eviction_count += 1
        
        logger.debug(f"Freed {freed_mb:.2f}MB from pool {pool.name}")
    
    async def _move_to_disk_overflow(self, pool: MemoryPool, key: str):
        """Move data to disk overflow"""
        if not pool.overflow_directory:
            return
        
        try:
            data = pool.data[key]
            overflow_file = pool.overflow_directory / f"{key}.overflow"
            
            # Serialize and save to disk
            with open(overflow_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Remove from memory
            del pool.data[key]
            
            # Track overflow operation
            self.performance_stats["overflow_operations"] += 1
            
            logger.debug(f"Moved {key} to disk overflow")
            
        except Exception as e:
            logger.error(f"Failed to move {key} to disk overflow: {e}")
    
    async def _estimate_data_size(self, data: Any) -> float:
        """Estimate the memory size of data in MB"""
        try:
            # Use sys.getsizeof as approximation
            size_bytes = sys.getsizeof(data)
            
            # For collections, recursively estimate
            if isinstance(data, (list, tuple)):
                for item in data:
                    size_bytes += sys.getsizeof(item)
            elif isinstance(data, dict):
                for key, value in data.items():
                    size_bytes += sys.getsizeof(key) + sys.getsizeof(value)
            
            return size_bytes / (1024 * 1024)  # Convert to MB
            
        except Exception:
            # Fallback estimation
            return 1.0  # 1MB default
    
    async def _compress_data(self, data: Any) -> Any:
        """Compress data to save memory"""
        try:
            import gzip
            import pickle
            
            # Serialize and compress
            serialized = pickle.dumps(data)
            compressed = gzip.compress(serialized)
            
            return compressed
            
        except Exception as e:
            logger.warning(f"Failed to compress data: {e}")
            return data
    
    async def _memory_monitoring_loop(self):
        """Background loop for memory monitoring"""
        while True:
            try:
                stats = await self._collect_memory_stats()
                self.memory_stats_history.append(stats)
                
                # Keep only recent history
                if len(self.memory_stats_history) > 1000:
                    self.memory_stats_history = self.memory_stats_history[-1000:]
                
                # Check memory pressure
                await self._check_memory_pressure(stats)
                
                # Sleep before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_memory_stats(self) -> MemoryStats:
        """Collect current memory statistics"""
        stats = MemoryStats(timestamp=datetime.now())
        
        try:
            # System memory
            memory_info = psutil.virtual_memory()
            stats.total_memory_mb = memory_info.total / (1024 * 1024)
            stats.available_memory_mb = memory_info.available / (1024 * 1024)
            stats.used_memory_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
            stats.memory_percent = memory_info.percent
            
            # Process memory
            process = psutil.Process()
            process_info = process.memory_info()
            stats.process_memory_mb = process_info.rss / (1024 * 1024)
            stats.process_memory_percent = (process_info.rss / memory_info.total) * 100
            
            # Pool usage
            for pool_name, pool in self.memory_pools.items():
                stats.pool_usage[pool_name] = pool.current_size_mb
            
            # Cache memory
            cache_pool = self.memory_pools.get(MemoryPoolType.CACHE.value)
            if cache_pool:
                stats.cache_memory_mb = cache_pool.current_size_mb
            
            # Peak memory (simplified tracking)
            if hasattr(self, '_peak_memory_mb'):
                stats.peak_memory_mb = self._peak_memory_mb
            else:
                self._peak_memory_mb = stats.process_memory_mb
                stats.peak_memory_mb = stats.process_memory_mb
            
            if stats.process_memory_mb > self._peak_memory_mb:
                self._peak_memory_mb = stats.process_memory_mb
                stats.peak_memory_mb = stats.process_memory_mb
            
            # Determine pressure level
            if stats.memory_percent >= self.memory_config["critical_threshold_percent"]:
                stats.pressure_level = MemoryPressureLevel.CRITICAL
            elif stats.memory_percent >= self.memory_config["warning_threshold_percent"]:
                stats.pressure_level = MemoryPressureLevel.HIGH
            elif stats.memory_percent >= 60:
                stats.pressure_level = MemoryPressureLevel.MODERATE
            else:
                stats.pressure_level = MemoryPressureLevel.LOW
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to collect memory stats: {e}")
            return stats
    
    async def _check_memory_pressure(self, stats: MemoryStats):
        """Check memory pressure and take appropriate actions"""
        if stats.pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
            logger.warning(f"High memory pressure detected: {stats.memory_percent:.1f}%")
            
            # Trigger memory optimization
            await self._handle_memory_pressure(stats.pressure_level)
            
            # Notify callbacks
            for callback in self.memory_pressure_callbacks:
                try:
                    await callback(stats.pressure_level)
                except Exception as e:
                    logger.error(f"Memory pressure callback error: {e}")
    
    async def _handle_memory_pressure(self, pressure_level: MemoryPressureLevel):
        """Handle memory pressure by freeing up memory"""
        logger.info(f"Handling {pressure_level.value} memory pressure")
        
        if pressure_level == MemoryPressureLevel.HIGH:
            # Moderate response
            await self._aggressive_cache_cleanup()
            await self._evict_old_large_documents()
            
        elif pressure_level == MemoryPressureLevel.CRITICAL:
            # Aggressive response
            await self._emergency_memory_cleanup()
            await self._force_garbage_collection()
        
        self.performance_stats["memory_optimizations"] += 1
    
    async def _aggressive_cache_cleanup(self):
        """Perform aggressive cache cleanup"""
        logger.info("Performing aggressive cache cleanup")
        
        # Reduce cache sizes by 50%
        for pool in self.memory_pools.values():
            if pool.pool_type == MemoryPoolType.CACHE:
                target_size = pool.current_size_mb * 0.5
                required_reduction = pool.current_size_mb - target_size
                
                if required_reduction > 0:
                    await self._free_pool_memory(pool, required_reduction)
    
    async def _evict_old_large_documents(self):
        """Evict old large documents"""
        logger.info("Evicting old large documents")
        
        current_time = datetime.now()
        eviction_threshold = timedelta(minutes=30)  # Evict documents not accessed in 30 minutes
        
        docs_to_evict = []
        for doc_id, handle in self.large_documents.items():
            if current_time - handle.last_accessed > eviction_threshold:
                docs_to_evict.append(doc_id)
        
        for doc_id in docs_to_evict:
            await self._evict_large_document(doc_id)
    
    async def _emergency_memory_cleanup(self):
        """Emergency memory cleanup for critical pressure"""
        logger.warning("Performing emergency memory cleanup")
        
        # Clear all non-essential caches
        non_essential_pools = [
            MemoryPoolType.TEMPORARY_DATA.value,
            MemoryPoolType.SEARCH_RESULTS.value
        ]
        
        for pool_name in non_essential_pools:
            if pool_name in self.memory_pools:
                pool = self.memory_pools[pool_name]
                pool.data.clear()
                pool.access_times.clear()
                pool.access_frequency.clear()
                pool.current_size_mb = 0
        
        # Evict all large documents
        for doc_id in list(self.active_large_docs):
            await self._evict_large_document(doc_id)
    
    async def _garbage_collection_loop(self):
        """Background garbage collection loop"""
        while True:
            try:
                await asyncio.sleep(self.memory_config["gc_frequency_seconds"])
                await self._force_garbage_collection()
                
            except Exception as e:
                logger.error(f"Garbage collection error: {e}")
                await asyncio.sleep(120)  # Wait longer on error
    
    async def _force_garbage_collection(self):
        """Force garbage collection"""
        logger.debug("Running garbage collection")
        
        # Run garbage collection
        collected = gc.collect()
        
        self.performance_stats["gc_collections"] += 1
        logger.debug(f"Garbage collection completed: {collected} objects collected")
    
    def register_memory_pressure_callback(self, callback: callable):
        """Register a callback for memory pressure events"""
        self.memory_pressure_callbacks.append(callback)
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        current_stats = await self._collect_memory_stats()
        
        # Calculate averages from history
        if self.memory_stats_history:
            recent_stats = self.memory_stats_history[-10:]  # Last 10 measurements
            avg_memory_percent = sum(s.memory_percent for s in recent_stats) / len(recent_stats)
            avg_process_memory = sum(s.process_memory_mb for s in recent_stats) / len(recent_stats)
        else:
            avg_memory_percent = current_stats.memory_percent
            avg_process_memory = current_stats.process_memory_mb
        
        # Pool statistics
        pool_stats = {}
        for pool_name, pool in self.memory_pools.items():
            pool_stats[pool_name] = {
                "current_size_mb": pool.current_size_mb,
                "max_size_mb": pool.max_size_mb,
                "utilization_percent": (pool.current_size_mb / pool.max_size_mb) * 100,
                "access_count": pool.access_count,
                "hit_count": pool.hit_count,
                "miss_count": pool.miss_count,
                "eviction_count": pool.eviction_count,
                "items_count": len(pool.data)
            }
        
        return {
            "current_memory": {
                "total_mb": current_stats.total_memory_mb,
                "available_mb": current_stats.available_memory_mb,
                "used_percent": current_stats.memory_percent,
                "process_mb": current_stats.process_memory_mb,
                "pressure_level": current_stats.pressure_level.value
            },
            "averages": {
                "memory_percent": avg_memory_percent,
                "process_memory_mb": avg_process_memory
            },
            "memory_pools": pool_stats,
            "large_documents": {
                "total_count": len(self.large_documents),
                "active_count": len(self.active_large_docs),
                "max_concurrent": self.memory_config["max_concurrent_large_docs"]
            },
            "performance_stats": self.performance_stats,
            "configuration": self.memory_config
        }
    
    async def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage and return optimization report"""
        logger.info("Starting memory optimization")
        
        optimization_report = {
            "timestamp": datetime.now().isoformat(),
            "actions_taken": [],
            "memory_freed_mb": 0.0,
            "optimization_success": True
        }
        
        try:
            initial_stats = await self._collect_memory_stats()
            
            # Optimize caches
            cache_freed = await self._optimize_caches()
            optimization_report["memory_freed_mb"] += cache_freed
            optimization_report["actions_taken"].append(f"Cache optimization freed {cache_freed:.2f}MB")
            
            # Optimize large document handling
            doc_freed = await self._optimize_large_documents()
            optimization_report["memory_freed_mb"] += doc_freed
            optimization_report["actions_taken"].append(f"Document optimization freed {doc_freed:.2f}MB")
            
            # Run garbage collection
            await self._force_garbage_collection()
            optimization_report["actions_taken"].append("Performed garbage collection")
            
            # Final stats
            final_stats = await self._collect_memory_stats()
            actual_freed = initial_stats.process_memory_mb - final_stats.process_memory_mb
            
            optimization_report["actual_memory_freed_mb"] = actual_freed
            optimization_report["initial_memory_mb"] = initial_stats.process_memory_mb
            optimization_report["final_memory_mb"] = final_stats.process_memory_mb
            
            logger.info(f"Memory optimization completed: {actual_freed:.2f}MB freed")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            optimization_report["optimization_success"] = False
            optimization_report["error"] = str(e)
        
        return optimization_report
    
    async def _optimize_caches(self) -> float:
        """Optimize cache usage and return memory freed"""
        memory_freed = 0.0
        
        for pool in self.memory_pools.values():
            if pool.pool_type == MemoryPoolType.CACHE:
                # Remove least accessed items
                initial_size = pool.current_size_mb
                
                # Free 25% of cache
                target_reduction = pool.current_size_mb * 0.25
                await self._free_pool_memory(pool, target_reduction)
                
                memory_freed += initial_size - pool.current_size_mb
        
        return memory_freed
    
    async def _optimize_large_documents(self) -> float:
        """Optimize large document memory usage"""
        memory_freed = 0.0
        
        # Evict documents that haven't been accessed recently
        current_time = datetime.now()
        threshold = timedelta(minutes=15)
        
        for doc_id in list(self.large_documents.keys()):
            handle = self.large_documents[doc_id]
            
            if current_time - handle.last_accessed > threshold:
                if handle.is_cached:
                    memory_freed += handle.total_size_mb
                
                await self._evict_large_document(doc_id)
        
        return memory_freed


# Global instance
memory_manager = MemoryManager()


# Compatibility methods for API
def get_memory_status(self) -> Dict[str, float]:
    """Get current memory status for API compatibility"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "process_mb": memory_info.rss / 1024 / 1024,
            "virtual_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    except:
        return {"process_mb": 0, "virtual_mb": 0, "percent": 0}

def optimize_memory(self) -> bool:
    """Simple memory optimization for API compatibility"""
    try:
        gc.collect()
        return True
    except:
        return False

# Add methods to MemoryManager class
MemoryManager.get_memory_status = get_memory_status
MemoryManager.optimize_memory = optimize_memory


# Utility functions
async def allocate_document_memory(
    document_id: str,
    estimated_size_mb: float,
    components: Dict[str, List[Any]]
) -> LargeDocumentHandle:
    """Allocate memory for document processing"""
    return await memory_manager.allocate_document_memory(document_id, estimated_size_mb, components)


async def get_memory_statistics() -> Dict[str, Any]:
    """Get current memory statistics"""
    return await memory_manager.get_memory_statistics()


async def optimize_memory_usage() -> Dict[str, Any]:
    """Optimize system memory usage"""
    return await memory_manager.optimize_memory_usage()


def register_memory_pressure_callback(callback: callable):
    """Register callback for memory pressure events"""
    memory_manager.register_memory_pressure_callback(callback)


if __name__ == "__main__":
    # Test the memory management system
    import asyncio
    
    async def test_memory_manager():
        await memory_manager.initialize()
        
        print("✅ Memory Manager initialized successfully")
        
        # Get current memory statistics
        stats = await get_memory_statistics()
        print(f"Current memory usage: {stats['current_memory']['process_mb']:.2f}MB")
        print(f"Memory pressure level: {stats['current_memory']['pressure_level']}")
        
        # Test document allocation
        test_components = {
            "sections": [{"id": "1", "content": "Test section"}],
            "chunks": [{"id": "1", "content": "Test chunk"} for _ in range(100)],
            "tables": [],
            "entities": [{"id": "1", "text": "Test entity"}]
        }
        
        handle = await allocate_document_memory("test_doc_001", 25.0, test_components)
        print(f"Allocated memory for document: {handle.document_id} ({handle.total_size_mb:.2f}MB)")
        print(f"Streaming enabled: {handle.is_streaming}")
        
        # Test memory optimization
        optimization_report = await optimize_memory_usage()
        print(f"Memory optimization completed:")
        print(f"  Actions taken: {len(optimization_report['actions_taken'])}")
        print(f"  Memory freed: {optimization_report['memory_freed_mb']:.2f}MB")
        
        # Get pool statistics
        for pool_name, pool_stats in stats['memory_pools'].items():
            print(f"Pool {pool_name}: {pool_stats['current_size_mb']:.2f}/{pool_stats['max_size_mb']:.2f}MB")
    
    asyncio.run(test_memory_manager())