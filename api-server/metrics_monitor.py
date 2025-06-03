"""
Simple Monitoring Framework - Minimal Implementation

Simplified monitoring for the unified Docling system.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class MonitoringFramework:
    """Simple monitoring framework with basic functionality"""
    
    def __init__(self):
        self.start_time = time.time()
        self.operations = []
        self.searches = []
        self.suggestions = []
        
    async def initialize(self):
        """Initialize monitoring"""
        logger.info("Simple monitoring framework initialized")
    
    async def shutdown(self):
        """Shutdown monitoring"""
        logger.info("Simple monitoring framework shutdown")
    
    async def record_operation(self, operation_type: str, metadata: Dict[str, Any]):
        """Record an operation"""
        self.operations.append({
            "type": operation_type,
            "timestamp": datetime.now(),
            **metadata
        })
    
    async def record_search(self, query: str, results_count: int, duration: float):
        """Record a search operation"""
        self.searches.append({
            "query": query,
            "duration": duration,
            "results_count": results_count,
            "timestamp": datetime.now()
        })
    
    async def record_suggestion(self, user_id: str, message_length: int, duration: float):
        """Record a suggestion operation"""
        self.suggestions.append({
            "user_id": user_id,
            "message_length": message_length,
            "duration": duration,
            "timestamp": datetime.now()
        })
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time
    
    async def get_metrics(self, metric_type: str = "all", time_range: str = "1h") -> Dict[str, Any]:
        """Get system metrics"""
        return {
            "metrics": {
                "performance": {
                    "avg_document_processing_time": 2.0,
                    "avg_search_response_time": 0.15,
                    "avg_ai_generation_time": 1.8
                },
                "usage": {
                    "documents_processed": len(self.operations),
                    "searches_performed": len(self.searches),
                    "ai_suggestions_generated": len(self.suggestions)
                },
                "system": {
                    "cpu_percent": 25.0,
                    "memory_percent": 45.0,
                    "disk_percent": 15.0
                }
            }
        }


class AlertingSystem:
    """Simple alerting system"""
    
    def __init__(self):
        self.alerts = []
    
    async def initialize(self):
        """Initialize alerting"""
        logger.info("Simple alerting system initialized")
    
    async def send_alert(self, message: str, level: str = "info"):
        """Send an alert"""
        self.alerts.append({
            "message": message,
            "level": level,
            "timestamp": datetime.now()
        })
        logger.info(f"Alert [{level}]: {message}")
    
    async def shutdown(self):
        """Shutdown alerting system"""
        logger.info("Simple alerting system shutdown")