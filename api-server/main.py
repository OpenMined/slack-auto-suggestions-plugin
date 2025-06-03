"""
Slack Robot API Server - Unified Docling-Only Implementation

This is the main entry point for the production API server using only
the unified Docling pipeline with no legacy code.
"""

import uvicorn
import logging
from pathlib import Path

# Import the unified Docling API
from api_server import app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_server.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the API server"""
    logger.info("Starting Slack Robot API Server (Unified Docling Implementation)")
    
    # Server configuration
    host = "0.0.0.0"
    port = 8000
    workers = 1  # Single worker for development, increase for production
    
    # Check if running in production mode
    import os
    is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    if is_production:
        logger.info("Running in PRODUCTION mode")
        # Production configuration
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            workers=4,  # Multiple workers for production
            log_level="info",
            access_log=True,
            use_colors=False,
            reload=False
        )
    else:
        logger.info("Running in DEVELOPMENT mode")
        # Development configuration
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            workers=workers,
            log_level="debug",
            access_log=True,
            use_colors=True,
            reload=True,  # Auto-reload on code changes
            reload_dirs=["./"]
        )

if __name__ == "__main__":
    main()