from typing import List, Dict
import numpy as np
from src.pipeline_manager import DocumentProcessingPipeline
from src.embedding_manager import EmbeddingManager
from src.logger import get_logger

logger = get_logger(__name__)

class EmbeddingPipeline:
    """
    Complete pipeline: Document Processing → Chunking → all-mpnet-base-v2 Embeddings
    Simple, focused, and production-ready
    """
    
    def __init__(self, 
                 chunking_strategy: str = "semantic",
                 chunk_size: int = 1000,  # Good balance for all-mpnet-base-v2
                 overlap: int = 200,
                 batch_size: int = 32):
        
        # Initialize document processing pipeline
        self.doc_pipeline = DocumentProcessingPipeline(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        # Initialize embedding manager (simplified)
        self.embedding_manager = EmbeddingManager()
        self.batch_size = batch_size
        
        logger.info(f"EmbeddingPipeline initialized:")
        logger.info(f"  Chunking: {chunking_strategy}, size={chunk_size}, overlap={overlap}")
        logger.info(f"  Embedding: {self.embedding_manager.model_name}")

    def process_documents_to_embeddings(self, file_paths: List[str]) -> Dict[str, Dict]:
        """
        Complete pipeline: Files → Chunks → Embeddings
        Returns ready-to-store data for vector database
        """
        logger.info(f"Starting embedding pipeline for {len(file_paths)} files")
        
        # Step 1: Process documents to chunks (your existing pipeline)
        chunk_results = self.doc_pipeline.process_multiple_documents(file_paths)
        
        # Step 2: Generate embeddings for all chunks
        embedding_results = {}
        
        for file_path, chunks in chunk_results.items():
            if not chunks:
                logger.warning(f"No chunks generated for {file_path}")
                embedding_results[file_path] = {
                    'chunks': [],
                    'embeddings': np.array([]),
                    'chunk_count': 0
                }
                continue
            
            logger.info(f"Processing {len(chunks)} chunks from {file_path}")
            
            # Extract chunk texts
            chunk_texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings using all-mpnet-base-v2
            embeddings = self.embedding_manager.generate_embeddings(
                chunk_texts, 
                batch_size=self.batch_size
            )
            
            # Store results
            embedding_results[file_path] = {
                'chunks': chunks,
                'embeddings': embeddings,
                'chunk_count': len(chunks),
                'embedding_dimension': self.embedding_manager.embedding_dim
            }
            
            logger.info(f"Generated {len(embeddings)} embeddings for {file_path}")
        
        total_chunks = sum(result['chunk_count'] for result in embedding_results.values())
        logger.info(f"Pipeline complete: {total_chunks} total embeddings from {len(file_paths)} files")
        
        return embedding_results

    def process_single_document(self, file_path: str) -> Dict:
        """Process single document"""
        results = self.process_documents_to_embeddings([file_path])
        return results[file_path]

    def search_similar_chunks(self, results: Dict, query: str, top_k: int = 5) -> List[tuple]:
        """
        Find most similar chunks to a query
        Returns: [(similarity_score, chunk_data), ...]
        """
        if not results['chunks']:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Compute similarities
        similarities = []
        for i, (chunk, embedding) in enumerate(zip(results['chunks'], results['embeddings'])):
            similarity = self.embedding_manager.compute_simiarity(query_embedding, embedding)
            similarities.append((similarity, chunk))
        
        # Return top-k most similar
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
