from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict, Tuple
from src.logger import get_logger

logger = get_logger(__name__)

class RerankerService:
    """
    Cross-encoder based reranking for improved retrieval relevance
    Uses ms-marco-MiniLM-L-6-v2 for fast, accurate reranking
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load cross-encoder reranking model"""
        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info(f"✅ Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise
    
    def rerank(self, query: str, documents: List[str], metadata: List[Dict], top_k: int = 5):
        """Rerank documents and return three separate values"""
        
        if not documents:
            logger.warning("No documents to rerank")
            return [], [], []
        
        logger.info(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")
        
        try:
            # Create query-document pairs
            query_doc_pairs = [(query, doc) for doc in documents]
            
            # Get reranking scores
            rerank_scores = self.model.predict(query_doc_pairs)
            
            # Convert to Python list if needed
            if hasattr(rerank_scores, 'tolist'):
                rerank_scores = rerank_scores.tolist()
            elif isinstance(rerank_scores, np.ndarray):
                rerank_scores = rerank_scores.tolist()
            
            # Ensure we have a list of floats
            rerank_scores = [float(score) for score in rerank_scores]
            
            # Create scored results and sort
            scored_results = list(zip(documents, metadata, rerank_scores))
            scored_results.sort(key=lambda x: x[2], reverse=True)  # Sort by score descending
            
            # Take top-k and unpack
            top_results = scored_results[:top_k]
            
            if not top_results:
                logger.warning("No results after reranking")
                return [], [], []
            
            reranked_docs = [item[0] for item in top_results]
            reranked_metadata = [item[1] for item in top_results]
            final_scores = [item[2] for item in top_results]
            
            logger.info(f"Reranking complete: {len(reranked_docs)} docs, top score = {max(final_scores):.3f}")
            
            return reranked_docs, reranked_metadata, final_scores
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order as fallback
            return documents[:top_k], metadata[:top_k], [0.5] * min(len(documents), top_k)