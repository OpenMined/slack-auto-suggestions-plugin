"""
Unified Docling Document Processor - Production Implementation

This is the single, unified document processor using only Docling for all document
processing. No legacy fallbacks, no dual processing, just pure Docling power.
"""

import asyncio
import hashlib
import logging
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

import psutil
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument, TableItem, TextItem, PictureItem
from sentence_transformers import SentenceTransformer

# Entity extraction support
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class UnifiedDocumentSection:
    """Represents a document section with hierarchy"""
    section_id: str
    title: str
    content: str
    level: int
    parent_id: Optional[str] = None
    reading_order: int = 0
    section_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedDocumentChunk:
    """Enhanced chunk with full context"""
    chunk_id: str
    content: str
    chunk_index: int
    section_id: str
    chunk_type: str = "text"
    
    # Context and relationships
    context_before: str = ""
    context_after: str = ""
    reading_order: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    token_count: int = 0


@dataclass
class UnifiedDocumentTable:
    """Structured table representation"""
    table_id: str
    section_id: str
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    table_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedDocumentEntity:
    """Extracted entity with context"""
    entity_id: str
    text: str
    entity_type: str
    confidence: float
    section_id: str
    chunk_id: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedProcessingResult:
    """Complete result from unified processing"""
    success: bool
    document_id: str
    filename: str
    
    # Extracted components
    sections: List[UnifiedDocumentSection] = field(default_factory=list)
    chunks: List[UnifiedDocumentChunk] = field(default_factory=list)
    tables: List[UnifiedDocumentTable] = field(default_factory=list)
    entities: List[UnifiedDocumentEntity] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class UnifiedDoclingProcessor:
    """
    Production-ready unified document processor using only Docling.
    No legacy code, no dual processing, just pure Docling functionality.
    """
    
    def __init__(self):
        # Core components
        self.converter = DocumentConverter()
        self.chunker = HybridChunker(
            tokenizer_model="all-MiniLM-L6-v2",
            max_tokens=512,
            overlap_tokens=50
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Entity extraction
        self.nlp_model = nlp if SPACY_AVAILABLE else None
        
        # Processing configuration
        self.config = {
            "max_file_size": 100 * 1024 * 1024,  # 100MB
            "chunk_size": 512,
            "chunk_overlap": 50,
            "min_chunk_size": 100,
            "extract_entities": True,
            "entity_confidence_threshold": 0.7,
            "generate_embeddings": True,
            "respect_section_boundaries": True
        }
        
        # Performance monitoring
        self.process = psutil.Process()
        
        logger.info(f"Unified Docling Processor initialized with config: {self.config}")
    
    async def process_text_file_directly(
        self,
        file_data: bytes,
        filename: str,
        user_id: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> UnifiedProcessingResult:
        """
        Process text files directly without Docling (for .txt files)
        
        Args:
            file_data: Raw file bytes
            filename: Original filename
            user_id: User who uploaded the document
            metadata: Additional metadata
            
        Returns:
            UnifiedProcessingResult with processed text
        """
        start_time = time.time()
        start_memory = self.process.memory_info().rss
        
        try:
            # Generate document ID
            document_id = hashlib.md5(file_data).hexdigest()[:16]
            
            logger.info(f"Processing text file directly: {filename}")
            
            # Decode text content
            try:
                text_content = file_data.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text_content = file_data.decode('latin-1')
                except UnicodeDecodeError:
                    return UnifiedProcessingResult(
                        success=False,
                        document_id=document_id,
                        filename=filename,
                        error="Unable to decode text file encoding"
                    )
            
            # Create simple sections (split by paragraphs)
            sections = await self._create_text_sections(text_content, document_id, filename)
            
            # Create simple chunks
            chunks = await self._create_text_chunks(text_content, sections, document_id)
            
            # Extract entities if enabled
            entities = []
            if self.config["extract_entities"]:
                entities = await self._extract_text_entities(text_content, document_id)
            
            # Generate embeddings
            if self.config["generate_embeddings"]:
                chunks = await self._generate_embeddings(chunks)
            
            # Calculate processing stats
            processing_time = time.time() - start_time
            memory_used = (self.process.memory_info().rss - start_memory) / 1024 / 1024  # MB
            
            # Build metadata
            result_metadata = {
                "filename": filename,
                "user_id": user_id,
                "processed_at": datetime.now().isoformat(),
                "processing_method": "direct_text",
                "processing_time": processing_time,
                "memory_used_mb": memory_used,
                **(metadata or {})
            }
            
            return UnifiedProcessingResult(
                success=True,
                document_id=document_id,
                filename=filename,
                sections=sections,
                chunks=chunks,
                tables=[],  # No tables in plain text
                entities=entities,
                processing_stats={
                    "sections_extracted": len(sections),
                    "chunks_created": len(chunks),
                    "tables_found": 0,
                    "entities_extracted": len(entities),
                    "processing_time_seconds": processing_time,
                    "memory_used_mb": memory_used
                },
                metadata=result_metadata
            )
            
        except Exception as e:
            logger.error(f"Error processing text file {filename}: {e}")
            return UnifiedProcessingResult(
                success=False,
                document_id=document_id,
                filename=filename,
                error=f"Text processing failed: {str(e)}"
            )
    
    async def process_document(
        self,
        file_data: bytes,
        filename: str,
        user_id: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> UnifiedProcessingResult:
        """
        Process a document through the unified Docling pipeline.
        
        Args:
            file_data: Raw file bytes
            filename: Original filename
            user_id: User who uploaded the document
            metadata: Additional metadata
            
        Returns:
            UnifiedProcessingResult with all extracted components
        """
        start_time = time.time()
        start_memory = self.process.memory_info().rss
        
        # Generate document ID
        document_id = self._generate_document_id(filename, user_id)
        
        logger.info(f"Processing document: {filename}", extra={
            "document_id": document_id,
            "user_id": user_id,
            "file_size": len(file_data)
        })
        
        try:
            # Validate input
            validation_result = self._validate_input(file_data, filename)
            if not validation_result["valid"]:
                return UnifiedProcessingResult(
                    success=False,
                    document_id=document_id,
                    filename=filename,
                    error=validation_result["error"]
                )
            
            # Process with Docling
            docling_result = await self._process_with_docling(file_data, filename)
            if not docling_result:
                return UnifiedProcessingResult(
                    success=False,
                    document_id=document_id,
                    filename=filename,
                    error="Docling processing failed"
                )
            
            # Extract structured components
            sections = await self._extract_sections(docling_result, document_id)
            chunks = await self._create_intelligent_chunks(docling_result, sections, document_id)
            tables = await self._extract_tables(docling_result, sections, document_id)
            entities = await self._extract_entities(sections, chunks, document_id)
            
            # Generate embeddings
            if self.config["generate_embeddings"]:
                chunks = await self._generate_embeddings(chunks)
            
            # Calculate processing stats
            processing_time = time.time() - start_time
            memory_used = (self.process.memory_info().rss - start_memory) / 1024 / 1024  # MB
            
            # Build metadata
            doc_metadata = {
                "filename": filename,
                "user_id": user_id,
                "processed_at": datetime.now().isoformat(),
                "docling_version": "latest",
                "processing_time": processing_time,
                "memory_used_mb": memory_used,
                **(metadata or {})
            }
            
            # Processing stats
            processing_stats = {
                "sections_extracted": len(sections),
                "chunks_created": len(chunks),
                "tables_found": len(tables),
                "entities_extracted": len(entities),
                "processing_time_seconds": processing_time,
                "memory_used_mb": memory_used
            }
            
            logger.info(f"Document processed successfully: {filename}", extra=processing_stats)
            
            return UnifiedProcessingResult(
                success=True,
                document_id=document_id,
                filename=filename,
                sections=sections,
                chunks=chunks,
                tables=tables,
                entities=entities,
                metadata=doc_metadata,
                processing_stats=processing_stats
            )
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            return UnifiedProcessingResult(
                success=False,
                document_id=document_id,
                filename=filename,
                error=str(e)
            )
    
    def _generate_document_id(self, filename: str, user_id: str) -> str:
        """Generate unique document ID"""
        content = f"{filename}_{user_id}_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _validate_input(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Validate input file"""
        # Check file size
        if len(file_data) > self.config["max_file_size"]:
            return {
                "valid": False,
                "error": f"File too large: {len(file_data) / 1024 / 1024:.1f}MB exceeds limit"
            }
        
        # Check file extension
        supported_extensions = {
            '.pdf', '.docx', '.doc', '.txt', '.md', '.rtf',
            '.xlsx', '.xls', '.pptx', '.ppt', '.html', '.xml'
        }
        
        file_ext = Path(filename).suffix.lower()
        if file_ext not in supported_extensions:
            return {
                "valid": False,
                "error": f"Unsupported file type: {file_ext}"
            }
        
        return {"valid": True}
    
    async def _process_with_docling(
        self,
        file_data: bytes,
        filename: str
    ) -> Optional[DoclingDocument]:
        """Process document with Docling converter"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                tmp_file.write(file_data)
                tmp_path = Path(tmp_file.name)
            
            # Convert with Docling
            result = self.converter.convert(tmp_path)
            
            # Clean up
            tmp_path.unlink()
            
            return result
            
        except Exception as e:
            logger.error(f"Docling conversion failed: {e}")
            return None
    
    async def _extract_sections(
        self,
        docling_doc: DoclingDocument,
        document_id: str
    ) -> List[UnifiedDocumentSection]:
        """Extract hierarchical sections from Docling document"""
        sections = []
        section_counter = 0
        
        # Build section hierarchy
        current_level = 0
        parent_stack = []
        
        for item in docling_doc.document.texts:
            if isinstance(item, TextItem):
                # Determine section level based on formatting or heuristics
                level = self._determine_section_level(item)
                
                # Adjust parent stack
                while len(parent_stack) > level:
                    parent_stack.pop()
                
                parent_id = parent_stack[-1] if parent_stack else None
                
                # Create section
                section_id = f"{document_id}_sec_{section_counter}"
                section = UnifiedDocumentSection(
                    section_id=section_id,
                    title=self._extract_section_title(item),
                    content=item.text,
                    level=level,
                    parent_id=parent_id,
                    reading_order=section_counter,
                    metadata={
                        "original_formatting": getattr(item, "formatting", {}),
                        "confidence": getattr(item, "confidence", 1.0)
                    }
                )
                
                sections.append(section)
                
                # Update parent stack
                if level >= len(parent_stack):
                    parent_stack.append(section_id)
                else:
                    parent_stack[level] = section_id
                
                section_counter += 1
        
        return sections
    
    async def _create_intelligent_chunks(
        self,
        docling_doc: DoclingDocument,
        sections: List[UnifiedDocumentSection],
        document_id: str
    ) -> List[UnifiedDocumentChunk]:
        """Create intelligent chunks respecting document structure"""
        chunks = []
        chunk_counter = 0
        
        # Use Docling's hybrid chunker on the entire document
        doc_chunks = self.chunker.chunk(docling_doc.document)
        
        for chunk in doc_chunks:
            # Skip empty chunks
            if not chunk.text.strip():
                continue
            
            # Create chunk with context
            chunk_id = f"{document_id}_chunk_{chunk_counter}"
            
            # Find which section this chunk belongs to (simple approximation)
            section_id = sections[0].section_id if sections else f"{document_id}_sec_0"
            section_title = sections[0].title if sections else "Document"
            
            chunk_obj = UnifiedDocumentChunk(
                chunk_id=chunk_id,
                content=chunk.text,
                chunk_index=chunk_counter,
                section_id=section_id,
                context_before="",  # Can be improved later
                context_after="",   # Can be improved later
                reading_order=chunk_counter,
                token_count=len(chunk.text.split()),
                metadata={
                    "section_title": section_title,
                    "section_level": 1,
                    "chunk_position": chunk_counter,
                    "total_chunks_in_document": "unknown"  # Will be updated after processing
                }
            )
            
            chunks.append(chunk_obj)
            chunk_counter += 1
        
        return chunks
    
    async def _extract_tables(
        self,
        docling_doc: DoclingDocument,
        sections: List[UnifiedDocumentSection],
        document_id: str
    ) -> List[UnifiedDocumentTable]:
        """Extract tables from Docling document"""
        tables = []
        table_counter = 0
        
        for item in docling_doc.document.tables:
            if isinstance(item, TableItem):
                # Find the section this table belongs to
                section_id = self._find_table_section(item, sections)
                
                # Extract table data
                table_id = f"{document_id}_table_{table_counter}"
                
                # Parse table structure
                headers, rows = self._parse_table_data(item)
                
                table = UnifiedDocumentTable(
                    table_id=table_id,
                    section_id=section_id or f"{document_id}_sec_0",
                    headers=headers,
                    rows=rows,
                    caption=getattr(item, "caption", None),
                    table_index=table_counter,
                    metadata={
                        "original_format": getattr(item, "format", "unknown"),
                        "confidence": getattr(item, "confidence", 1.0)
                    }
                )
                
                tables.append(table)
                table_counter += 1
        
        return tables
    
    async def _extract_entities(
        self,
        sections: List[UnifiedDocumentSection],
        chunks: List[UnifiedDocumentChunk],
        document_id: str
    ) -> List[UnifiedDocumentEntity]:
        """Extract named entities from document content"""
        entities = []
        entity_counter = 0
        
        if not self.config["extract_entities"]:
            return entities
        
        # Extract from sections
        for section in sections:
            section_entities = self._extract_entities_from_text(
                section.content,
                section.section_id,
                None,
                entity_counter
            )
            entities.extend(section_entities)
            entity_counter += len(section_entities)
        
        # Extract from chunks (avoiding duplicates)
        seen_entities = {(e.text, e.entity_type) for e in entities}
        
        for chunk in chunks:
            chunk_entities = self._extract_entities_from_text(
                chunk.content,
                chunk.section_id,
                chunk.chunk_id,
                entity_counter
            )
            
            # Filter out duplicates
            for entity in chunk_entities:
                if (entity.text, entity.entity_type) not in seen_entities:
                    entities.append(entity)
                    seen_entities.add((entity.text, entity.entity_type))
                    entity_counter += 1
        
        return entities
    
    def _extract_entities_from_text(
        self,
        text: str,
        section_id: str,
        chunk_id: Optional[str],
        start_counter: int
    ) -> List[UnifiedDocumentEntity]:
        """Extract entities from text using spaCy or fallback patterns"""
        entities = []
        
        if self.nlp_model and SPACY_AVAILABLE:
            # Use spaCy for entity extraction
            doc = self.nlp_model(text)
            
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "MONEY", "PRODUCT"]:
                    entity_id = f"entity_{start_counter + len(entities)}"
                    
                    entity = UnifiedDocumentEntity(
                        entity_id=entity_id,
                        text=ent.text,
                        entity_type=ent.label_,
                        confidence=0.9,  # spaCy doesn't provide confidence
                        section_id=section_id,
                        chunk_id=chunk_id,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        metadata={"source": "spacy"}
                    )
                    
                    entities.append(entity)
        else:
            # Fallback pattern-based extraction
            patterns = {
                "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "URL": r'https?://(?:www\.)?[\w\-\.]+(?:\.[\w\-]+)+[\w\-\._~:/?#[\]@!\$&\'\(\)\*\+,;=.]+',
                "PHONE": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                "DATE": r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b',
            }
            
            for entity_type, pattern in patterns.items():
                for match in re.finditer(pattern, text):
                    entity_id = f"entity_{start_counter + len(entities)}"
                    
                    entity = UnifiedDocumentEntity(
                        entity_id=entity_id,
                        text=match.group(),
                        entity_type=entity_type,
                        confidence=0.8,
                        section_id=section_id,
                        chunk_id=chunk_id,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        metadata={"source": "pattern"}
                    )
                    
                    entities.append(entity)
        
        return entities
    
    async def _generate_embeddings(
        self,
        chunks: List[UnifiedDocumentChunk]
    ) -> List[UnifiedDocumentChunk]:
        """Generate embeddings for chunks"""
        # Batch process for efficiency
        batch_size = 32
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Assign embeddings
            for j, chunk in enumerate(batch):
                chunk.embedding = embeddings[j].tolist()
        
        return chunks
    
    def _determine_section_level(self, item: TextItem) -> int:
        """Determine section hierarchy level from text item"""
        # Simple heuristic based on formatting
        # In production, this would use more sophisticated analysis
        text = item.text.strip()
        
        # Check for heading patterns
        if text.isupper() and len(text) < 100:
            return 0  # Top level
        elif text.endswith(':') and len(text) < 50:
            return 1  # Sub-section
        else:
            return 2  # Content
    
    def _extract_section_title(self, item: TextItem) -> str:
        """Extract section title from text item"""
        text = item.text.strip()
        
        # Use first line as title for sections
        lines = text.split('\n')
        title = lines[0].strip()
        
        # Clean up title
        if len(title) > 100:
            title = title[:97] + "..."
        
        return title or "Untitled Section"
    
    def _find_table_section(
        self,
        table_item: TableItem,
        sections: List[UnifiedDocumentSection]
    ) -> Optional[str]:
        """Find which section a table belongs to"""
        # This is a simplified implementation
        # In production, would use position/context information
        if sections:
            return sections[-1].section_id
        return None
    
    def _parse_table_data(self, table_item: TableItem) -> Tuple[List[str], List[List[str]]]:
        """Parse table data from Docling table item"""
        headers = []
        rows = []
        
        try:
            # Use Docling's built-in export_to_dataframe method
            df = table_item.export_to_dataframe()
            if df is not None and not df.empty:
                headers = df.columns.tolist()
                rows = df.values.tolist()
        except Exception as e:
            logger.warning(f"Failed to export table to dataframe: {e}")
            
            # Fallback: try to get table data using alternative methods
            try:
                # Try to export to markdown and parse (simple fallback)
                markdown_text = table_item.export_to_markdown()
                if markdown_text:
                    lines = markdown_text.strip().split('\n')
                    if len(lines) >= 2:
                        # Parse markdown table
                        header_line = lines[0].strip('|').split('|')
                        headers = [col.strip() for col in header_line if col.strip()]
                        
                        # Skip separator line and parse data rows
                        for line in lines[2:]:
                            if line.strip():
                                row_data = line.strip('|').split('|')
                                row = [cell.strip() for cell in row_data if cell.strip() or True]  # Keep empty cells
                                if row:
                                    rows.append(row)
            except Exception as fallback_error:
                logger.warning(f"Table extraction fallback also failed: {fallback_error}")
                # Return minimal table structure
                headers = ["Column"]
                rows = [["Table data could not be extracted"]]
        
        return headers, rows


# Global processor instance
_processor = None


def get_unified_processor() -> UnifiedDoclingProcessor:
    """Get global unified processor instance"""
    global _processor
    if _processor is None:
        _processor = UnifiedDoclingProcessor()
    return _processor


# Test function
if __name__ == "__main__":
    async def test_processor():
        processor = UnifiedDoclingProcessor()
        
        # Test with a simple text file
        test_content = b"""# Test Document
        
This is a test document for the unified processor.

## Section 1: Introduction
This section contains introductory content about the document processor.

## Section 2: Features
- Feature 1: Intelligent chunking
- Feature 2: Entity extraction
- Feature 3: Table parsing

### Subsection 2.1: Technical Details
More detailed technical information goes here.

## Section 3: Conclusion
Final thoughts and summary.
"""
        
        result = await processor.process_document(
            test_content,
            "test_document.md",
            "test_user"
        )
        
        print(f"Processing success: {result.success}")
        print(f"Sections extracted: {len(result.sections)}")
        print(f"Chunks created: {len(result.chunks)}")
        print(f"Entities found: {len(result.entities)}")
        print(f"Processing stats: {result.processing_stats}")
        
    # Run test
    import asyncio
    asyncio.run(test_processor())
    async def _create_text_sections(
        self, 
        text_content: str, 
        document_id: str, 
        filename: str
    ) -> List[UnifiedDocumentSection]:
        """Create sections from plain text by splitting on double newlines"""
        sections = []
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # If no double newlines, treat as one section
            paragraphs = [text_content.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            section_id = f"{document_id}_sec_{i}"
            
            # Try to extract a title from first line if it looks like a header
            lines = paragraph.split('\n')
            if len(lines) > 1 and len(lines[0]) < 100 and not lines[0].endswith('.'):
                title = lines[0].strip()
                content = '\n'.join(lines[1:]).strip()
                level = 1
            else:
                title = f"Section {i + 1}" if len(paragraphs) > 1 else filename
                content = paragraph
                level = 2 if len(paragraphs) > 1 else 1
            
            section = UnifiedDocumentSection(
                section_id=section_id,
                title=title,
                content=content,
                level=level,
                reading_order=i,
                section_type="paragraph",
                metadata={
                    "paragraph_index": i,
                    "total_paragraphs": len(paragraphs),
                    "character_count": len(content)
                }
            )
            sections.append(section)
        
        return sections
    
    async def _create_text_chunks(
        self, 
        text_content: str, 
        sections: List[UnifiedDocumentSection], 
        document_id: str
    ) -> List[UnifiedDocumentChunk]:
        """Create chunks from plain text using simple text splitting"""
        chunks = []
        chunk_counter = 0
        
        for section in sections:
            # Simple chunking by splitting on sentences or fixed size
            section_text = section.content
            
            # Split into sentences first
            sentences = re.split(r'[.\!?]+\s+', section_text)
            
            current_chunk = ""
            for sentence in sentences:
                # If adding this sentence would exceed chunk size, create a chunk
                if (len(current_chunk) + len(sentence) > self.config["chunk_size"] 
                    and current_chunk.strip()):
                    
                    chunk_id = f"{document_id}_chunk_{chunk_counter}"
                    chunk = UnifiedDocumentChunk(
                        chunk_id=chunk_id,
                        content=current_chunk.strip(),
                        chunk_index=chunk_counter,
                        section_id=section.section_id,
                        context_before="",
                        context_after="",
                        reading_order=chunk_counter,
                        token_count=len(current_chunk.split()),
                        metadata={
                            "section_title": section.title,
                            "section_level": section.level,
                            "chunk_method": "sentence_split"
                        }
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                    current_chunk = sentence + ". "
                else:
                    current_chunk += sentence + ". "
            
            # Add the remaining content as a chunk
            if current_chunk.strip():
                chunk_id = f"{document_id}_chunk_{chunk_counter}"
                chunk = UnifiedDocumentChunk(
                    chunk_id=chunk_id,
                    content=current_chunk.strip(),
                    chunk_index=chunk_counter,
                    section_id=section.section_id,
                    context_before="",
                    context_after="",
                    reading_order=chunk_counter,
                    token_count=len(current_chunk.split()),
                    metadata={
                        "section_title": section.title,
                        "section_level": section.level,
                        "chunk_method": "sentence_split"
                    }
                )
                chunks.append(chunk)
                chunk_counter += 1
        
        return chunks
    
    async def _extract_text_entities(
        self, 
        text_content: str, 
        document_id: str
    ) -> List[UnifiedDocumentEntity]:
        """Extract entities from plain text"""
        entities = []
        
        if self.has_spacy:
            # Use spaCy for entity extraction
            doc = self.nlp(text_content)
            entity_counter = 0
            
            for ent in doc.ents:
                if ent.label_ and len(ent.text.strip()) > 1:
                    entity_id = f"{document_id}_entity_{entity_counter}"
                    entity = UnifiedDocumentEntity(
                        entity_id=entity_id,
                        text=ent.text,
                        entity_type=ent.label_,
                        confidence=1.0,  # spaCy doesn't provide confidence scores directly
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        context="",  # Could be improved
                        metadata={
                            "extraction_method": "spacy",
                            "spacy_label": ent.label_,
                            "text_length": len(ent.text)
                        }
                    )
                    entities.append(entity)
                    entity_counter += 1
        else:
            # Fallback: simple pattern matching
            patterns = {
                "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z < /dev/null | a-z]{2,}\b',
                "URL": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[\!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                "DATE": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
                "PHONE": r'[\+]?[1-9]?[\d\s\-\(\)]{10,}',
                "MONEY": r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
            }
            
            entity_counter = 0
            for entity_type, pattern in patterns.items():
                for match in re.finditer(pattern, text_content):
                    entity_id = f"{document_id}_entity_{entity_counter}"
                    entity = UnifiedDocumentEntity(
                        entity_id=entity_id,
                        text=match.group(),
                        entity_type=entity_type,
                        confidence=0.8,  # Pattern matching confidence
                        start_char=match.start(),
                        end_char=match.end(),
                        context="",
                        metadata={
                            "extraction_method": "pattern_matching",
                            "pattern": pattern
                        }
                    )
                    entities.append(entity)
                    entity_counter += 1
        
        return entities
