from typing import Dict, List
from src.file_processor import DocumentProcessor
from src.chunker import Chunker  
from src.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessingPipeline:
    """
    Orchestrates the complete document processing pipeline:
    File Processing → Token-Aware Chunking
    """
    
    def __init__(self, strategy="semantic", chunk_size=1000, overlap=200, model_name="gpt-3.5-turbo"):
        self.document_processor = DocumentProcessor()
        # Use your token-aware chunker directly
        self.chunker = Chunker(
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            model_name=model_name
        )
        logger.info(f"Document processing pipeline initialized with token-aware chunker")

    def process_document_to_chunks(self, file_path: str) -> List[Dict]:
        """
        Complete pipeline: Process file → Create token-aware chunks
        """
        logger.info(f"Starting complete pipeline for: {file_path}")
        
        try:
            # Step 1: Process document with page information
            doc_data = self.document_processor.process_document(file_path)
            content = doc_data['content']
            metadata = doc_data['metadata']
            pages = doc_data.get('pages')
            
            logger.info(f"Document processed: {len(content)} characters, {len(pages) if pages else 'no'} pages")
            
            # Step 2: Create chunks with page awareness
            chunks = self.chunker.chunk_text(content, metadata, pages)
            
            logger.info(f"Pipeline complete: {len(chunks)} chunks created for {metadata['filename']}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Pipeline failed for {file_path}: {str(e)}")
            raise
    
    def process_multiple_documents(self, file_paths: List[str]) -> Dict[str, List[Dict]]:
        """Process multiple documents through the pipeline"""
        results = {}
        
        for file_path in file_paths:
            try:
                chunks = self.process_document_to_chunks(file_path)
                results[file_path] = chunks
                logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                results[file_path] = []
        
        total_chunks = sum(len(chunks) for chunks in results.values())
        logger.info(f"Batch processing complete: {total_chunks} total chunks from {len(file_paths)} files")
        
        return results
