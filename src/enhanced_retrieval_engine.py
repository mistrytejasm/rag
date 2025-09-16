from typing import List, Dict, Optional, Tuple
import numpy as np
from src.vector_store import VectorStore
from src.embedding_manager import EmbeddingManager
from src.reranker import RerankerService
from src.logger import get_logger

logger = get_logger(__name__)

class EnhancedRetrievalEngine:
    """
    Advanced retrieval engine with optional reranking for A/B testing
    Supports both standard retrieval and reranked retrieval
    """
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedding_manager: EmbeddingManager,
                 enable_reranking: bool = True):
        
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.enable_reranking = enable_reranking
        
        # Initialize reranker if enabled
        if self.enable_reranking:
            try:
                self.reranker = RerankerService()
                logger.info("Enhanced retrieval with reranking enabled")
            except Exception as e:
                logger.warning(f"Reranker failed to load, disabling: {e}")
                self.enable_reranking = False
                self.reranker = None
        else:
            self.reranker = None
            logger.info("Enhanced retrieval without reranking (A/B test group)")

    def retrieve_context(self, 
                        query: str,
                        top_k: int = 5,
                        fetch_k: Optional[int] = None,
                        filters: Optional[Dict] = None,
                        min_similarity: float = 0.0) -> Dict:
        """
        Retrieve context with optional reranking
        
        Args:
            query: User query
            top_k: Final number of results to return
            fetch_k: Number of initial results to fetch (for reranking)
            filters: Optional metadata filters
            min_similarity: Minimum similarity threshold
            
        Returns:
            Enhanced results with reranking metrics
        """
        
        # Determine fetch size based on reranking
        if self.enable_reranking and fetch_k is None:
            fetch_k = min(top_k * 4, 20)  # Fetch 4x more for reranking
        else:
            fetch_k = fetch_k or top_k
        
        logger.info(f"Retrieving context: query='{query[:50]}...', fetch_k={fetch_k}, final_k={top_k}")
        
        try:
            # Step 1: Generate query embedding
            query_embedding = self.embedding_manager.generate_embeddings([query])[0]
            
            # Step 2: Initial similarity search
            search_results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                n_results=fetch_k,
                filters=filters
            )
            
            if search_results['count'] == 0:
                logger.warning("No results found for query")
                return self._empty_results()
            
            # Step 3: Filter by minimum similarity
            filtered_results = self._filter_by_similarity(search_results, min_similarity)
            
            if not filtered_results['documents']:
                logger.warning(f"No results above similarity threshold {min_similarity}")
                return self._empty_results()
            
            # Step 4: Apply reranking if enabled
            if self.enable_reranking and self.reranker:
                reranked_results = self._apply_reranking(
                    query, filtered_results, top_k
                )
                return self._build_final_context(reranked_results, reranking_used=True)
            else:
                # No reranking: just take top_k from similarity results
                final_results = {
                    'documents': filtered_results['documents'][:top_k],
                    'metadata': filtered_results['metadata'][:top_k],
                    'distances': filtered_results['distances'][:top_k],
                    'ids': filtered_results['ids'][:top_k],
                    'scores': [1 - d for d in filtered_results['distances'][:top_k]]
                }
                return self._build_final_context(final_results, reranking_used=False)
                
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            raise

    def _apply_reranking(self, query: str, filtered_results: Dict, top_k: int) -> Dict:
        """Apply reranking to filtered results"""
        
        try:
            # Call reranker - it returns three separate values
            reranked_docs, reranked_metadata, rerank_scores = self.reranker.rerank(
                query=query,
                documents=filtered_results['documents'],
                metadata=filtered_results['metadata'],
                top_k=top_k
            )
            
            # Ensure we have valid results
            if not reranked_docs or not rerank_scores:
                logger.warning("Reranking returned empty results, falling back to original")
                return {
                    'documents': filtered_results['documents'][:top_k],
                    'metadata': filtered_results['metadata'][:top_k],
                    'scores': [1 - d for d in filtered_results['distances'][:top_k]],
                    'distances': filtered_results['distances'][:top_k],
                    'ids': filtered_results['ids'][:top_k]
                }
            
            return {
                'documents': reranked_docs,
                'metadata': reranked_metadata,
                'scores': rerank_scores,
                'distances': [1 - score for score in rerank_scores],
                'ids': [meta.get('chunk_id', f'chunk_{i}') for i, meta in enumerate(reranked_metadata)]
            }
            
        except Exception as e:
            logger.error(f"Reranking application failed: {e}")
            # Fallback to original results
            return {
                'documents': filtered_results['documents'][:top_k],
                'metadata': filtered_results['metadata'][:top_k], 
                'scores': [1 - d for d in filtered_results['distances'][:top_k]],
                'distances': filtered_results['distances'][:top_k],
                'ids': filtered_results['ids'][:top_k]
            }


    def _build_final_context(self, results: Dict, reranking_used: bool) -> Dict:
        """Build final context with citations and metrics"""
        
        if not results['documents']:
            return self._empty_results()
        
        # Build context string with citations
        context_parts = []
        citations = []
        
        for i, (doc, metadata, score) in enumerate(zip(
            results['documents'], 
            results['metadata'], 
            results['scores']
        )):
            citation_id = i + 1
            
            # Context with citation marker
            context_parts.append(f"[{citation_id}] {doc}")
            
            # Citation info
            citation = {
                'id': citation_id,
                'document_id': metadata.get('chunk_id', f'doc_{i}'),
                'similarity_score': round(float(score), 3),
                'source_file': metadata.get('source_file', 'Unknown'),
                'filename': metadata.get('filename', 'Unknown'),
                'page_number': metadata.get('start_page_number'),
                'chunk_index': metadata.get('chunk_index', i),
                'content_preview': doc[:150] + "..." if len(doc) > 150 else doc,
                'reranked': reranking_used
            }
            citations.append(citation)
        
        full_context = "\n\n".join(context_parts)
        
        return {
            'context': full_context,
            'citations': citations,
            'chunk_count': len(results['documents']),
            'avg_similarity': round(sum(results['scores']) / len(results['scores']), 3),
            'reranking_used': reranking_used,
            'retrieval_method': 'reranked' if reranking_used else 'similarity_only'
        }

    def _filter_by_similarity(self, search_results: Dict, min_similarity: float) -> Dict:
        """Filter results by minimum similarity threshold"""
        filtered_docs = []
        filtered_distances = []
        filtered_metadata = []
        filtered_ids = []
        
        for i, distance in enumerate(search_results['distances']):
            similarity = 1 - distance
            if similarity >= min_similarity:
                filtered_docs.append(search_results['documents'][i])
                filtered_distances.append(distance)
                filtered_metadata.append(search_results['metadata'][i])
                filtered_ids.append(search_results['ids'][i])
        
        return {
            'documents': filtered_docs,
            'distances': filtered_distances,
            'metadata': filtered_metadata,
            'ids': filtered_ids,
            'count': len(filtered_docs)
        }

    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'context': '',
            'citations': [],
            'chunk_count': 0,
            'avg_similarity': 0.0,
            'reranking_used': False,
            'retrieval_method': 'none'
        }
