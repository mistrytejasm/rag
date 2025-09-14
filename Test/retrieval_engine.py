from typing import List, Dict, Optional, Tuple
import numpy as np
from src.vector_store import VectorStore
from src.embedding_manager import EmbeddingManager
from src.logger import get_logger

logger = get_logger(__name__)

class RetrievalEngine:
    """
    Advanced retrieval engine with filtering, ranking, and citation support
    Handles complex queries and provides context for RAG generation
    """
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedding_manager: EmbeddingManager):
        
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        
        logger.info("RetrievalEngine initialized")

    def retrieve_context(self, 
                        query: str,
                        top_k: int = 5,
                        filters: Optional[Dict] = None,
                        min_similarity: float = 0.1) -> Dict:
        """
        Retrieve relevant context for a query with advanced filtering
        
        Args:
            query: User query string
            top_k: Number of top results to return
            filters: Optional metadata filters
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary with retrieved context and metadata
        """
        logger.info(f"Retrieving context for query: '{query}' (top_k={top_k})")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_embeddings([query])[0]
            
            # Perform similarity search
            search_results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                n_results=top_k * 2,  # Get more results for filtering
                filters=filters
            )
            
            if search_results['count'] == 0:
                logger.warning("No results found for query")
                return self._empty_results()
            
            # Filter by minimum similarity and rank results
            filtered_results = self._filter_and_rank_results(
                search_results, 
                min_similarity=min_similarity,
                top_k=top_k
            )
            
            # Build context with citations
            context = self._build_context_with_citations(filtered_results)
            
            logger.info(f"Retrieved {len(filtered_results['documents'])} relevant chunks")
            return context
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            raise

    def retrieve_by_page(self, 
                        query: str,
                        page_number: int,
                        source_file: Optional[str] = None,
                        top_k: int = 3) -> Dict:
        """Retrieve context from a specific page"""
        
        filters = {"start_page_number": page_number}
        if source_file:
            filters["source_file"] = source_file
        
        logger.info(f"Retrieving from page {page_number}" + (f" in {source_file}" if source_file else ""))
        
        return self.retrieve_context(
            query=query,
            top_k=top_k,
            filters=filters,
            min_similarity=0.2  # Lower threshold for page-specific search
        )

    def retrieve_by_source(self, 
                          query: str,
                          source_file: str,
                          top_k: int = 5) -> Dict:
        """Retrieve context from a specific source file"""
        
        filters = {"source_file": source_file}
        
        logger.info(f"Retrieving from source: {source_file}")
        
        return self.retrieve_context(
            query=query,
            top_k=top_k,
            filters=filters
        )

    def _filter_and_rank_results(self, 
                                search_results: Dict,
                                min_similarity: float,
                                top_k: int) -> Dict:
        """Filter results by similarity threshold and rank them"""
        
        filtered_docs = []
        filtered_distances = []
        filtered_metadata = []
        filtered_ids = []
        
        for i, distance in enumerate(search_results['distances']):
            # Convert distance to similarity (ChromaDB uses distance, lower = more similar)
            similarity = 1 - distance
            
            if similarity >= min_similarity:
                filtered_docs.append(search_results['documents'][i])
                filtered_distances.append(distance)
                filtered_metadata.append(search_results['metadata'][i])
                filtered_ids.append(search_results['ids'][i])
        
        # Take only top_k results
        return {
            'documents': filtered_docs[:top_k],
            'distances': filtered_distances[:top_k],
            'metadata': filtered_metadata[:top_k],
            'ids': filtered_ids[:top_k],
            'count': min(len(filtered_docs), top_k)
        }

    def _build_context_with_citations(self, results: Dict) -> Dict:
        """Build context string with proper citations"""
        
        if results['count'] == 0:
            return self._empty_results()
        
        context_parts = []
        citations = []
        
        for i, (doc, distance, metadata, doc_id) in enumerate(zip(
            results['documents'], 
            results['distances'], 
            results['metadata'], 
            results['ids']
        )):
            similarity = 1 - distance
            citation_id = i + 1
            
            # Build context with citation marker
            context_part = f"[{citation_id}] {doc}"
            context_parts.append(context_part)
            
            # Build citation information
            citation = {
                'id': citation_id,
                'document_id': doc_id,
                'similarity_score': round(similarity, 3),
                'source_file': metadata.get('source_file', 'Unknown'),
                'filename': metadata.get('filename', 'Unknown'),
                'page_number': metadata.get('start_page_number'),
                'chunk_index': metadata.get('chunk_index', 0),
                'content_preview': metadata.get('content_preview', doc + "..."),
                'spans_pages': metadata.get('spans_multiple_pages', False)
            }
            
            if citation['spans_pages']:
                citation['page_range'] = f"{metadata.get('start_page_number')}-{metadata.get('end_page_number')}"
            
            citations.append(citation)
        
        # Combine all context
        full_context = "\n\n".join(context_parts)
        
        return {
            'context': full_context,
            'citations': citations,
            'chunk_count': results['count'],
            'avg_similarity': round(sum(1 - d for d in results['distances']) / results['count'], 3)
        }

    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'context': '',
            'citations': [],
            'chunk_count': 0,
            'avg_similarity': 0.0
        }

    def get_similar_chunks_with_context(self, 
                                      chunk_id: str,
                                      top_k: int = 3) -> Dict:
        """Find chunks similar to a given chunk (for expanding context)"""
        
        # Get the original chunk
        original_chunk = self.vector_store.get_document_by_id(chunk_id)
        if not original_chunk:
            logger.warning(f"Chunk {chunk_id} not found")
            return self._empty_results()
        
        # Use chunk content as query
        return self.retrieve_context(
            query=original_chunk['content'],
            top_k=top_k + 1,  # +1 to account for the original chunk
            min_similarity=0.4
        )
