import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import uuid
import json
from datetime import datetime
from src.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    """
    ChromaDB-based vector store with rich metadata support
    Handles document storage, retrieval, and management for RAG system
    """
    
    def __init__(self, 
                 collection_name: str = "rag_documents",
                 persist_directory: str = "./vector_db"):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"VectorStore initialized: collection='{collection_name}', path='{persist_directory}'")

    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            doc_count = collection.count()
            logger.info(f"Found existing collection '{self.collection_name}' with {doc_count} documents")
            return collection
            
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document embeddings with page tracking"}
            )
            logger.info(f"Created new collection '{self.collection_name}'")
            return collection

    def add_documents(self, 
                    chunks: List[Dict], 
                    embeddings: np.ndarray,
                    source_file: str) -> List[str]:
        """
        Add document chunks with embeddings to vector store
        """
        if len(chunks) != len(embeddings):
                raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch")
    
        logger.info(f"Adding {len(chunks)} chunks from {source_file} to vector store")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Generate unique ID
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            
            # Extract content
            documents.append(chunk['content'])
            
            # Enhanced metadata for better retrieval and citations
            enhanced_metadata = {
                **chunk['metadata'],
                'chunk_id': chunk_id,
                'source_file': source_file,
                'added_timestamp': datetime.now().isoformat(),
                'embedding_model': 'all-mpnet-base-v2',
                'content_preview': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                'word_count': len(chunk['content'].split()),
                'has_page_info': 'start_page_number' in chunk['metadata']
            }
            
            # Remove None values
            cleaned_metadata = self._clean_metadata(enhanced_metadata)
            metadatas.append(cleaned_metadata)
            embeddings_list.append(embedding.tolist())
        
        # Add to ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings_list
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")

            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise

    def _clean_metadata(self, metadata: Dict) -> Dict:
        """Remove None values and ensure ChromaDB compatibility"""
        cleaned = {}
        
        for key, value in metadata.items():
            if value is None:
                if key in ['start_page_number', 'end_page_number', 'chunk_index', 'chunk_token_count']:
                    cleaned[key] = 0
                elif key in ['has_page_info', 'spans_multiple_pages']:
                    cleaned[key] = False
                else:
                    cleaned[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        
        return cleaned

        
    def similarity_search(self, 
                         query_embedding: np.ndarray,
                         n_results: int = 10,
                         filters: Optional[Dict] = None) -> Dict:
        """
        Search for similar documents using vector similarity
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            Dictionary with documents, distances, metadata, and IDs
        """
        try:
            # Convert numpy array to list for ChromaDB
            query_embedding_list = query_embedding.tolist()
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=n_results,
                where=filters,
                include=['documents', 'distances', 'metadatas']
            )
            
            # Process results
            if not results['documents'][0]:
                logger.warning("No results found for similarity search")
                return {
                    'documents': [],
                    'distances': [],
                    'metadata': [],
                    'ids': [],
                    'count': 0
                }
            
            search_results = {
                'documents': results['documents'][0],
                'distances': results['distances'][0],
                'metadata': results['metadatas'][0],
                'ids': results['ids'][0],
                'count': len(results['documents'][0])
            }
            
            logger.info(f"Found {search_results['count']} similar documents")
            return search_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise

    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a specific document by ID"""
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas']
            )
            
            if results['documents']:
                return {
                    'id': doc_id,
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    def delete_documents_by_source(self, source_file: str) -> int:
        """Delete all documents from a specific source file"""
        try:
            # Find documents from source file
            results = self.collection.get(
                where={"source_file": source_file},
                include=['metadatas']
            )
            
            if not results['ids']:
                logger.info(f"No documents found for source file: {source_file}")
                return 0
            
            # Delete documents
            self.collection.delete(ids=results['ids'])
            
            deleted_count = len(results['ids'])
            logger.info(f"Deleted {deleted_count} documents from {source_file}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete documents from {source_file}: {e}")
            raise

    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store"""
        try:
            total_docs = self.collection.count()
            
            # Get sample of metadata to analyze
            sample_results = self.collection.get(
                limit=min(100, total_docs),
                include=['metadatas']
            )
            
            # Analyze source files
            source_files = set()
            page_counts = []
            
            for metadata in sample_results['metadatas']:
                source_files.add(metadata.get('source_file', 'unknown'))
                if metadata.get('has_page_info'):
                    page_counts.append(metadata.get('start_page_number', 0))
            
            return {
                'total_documents': total_docs,
                'unique_source_files': len(source_files),
                'source_files': list(source_files),
                'has_page_tracking': len(page_counts) > 0,
                'page_range': f"{min(page_counts)}-{max(page_counts)}" if page_counts else "N/A"
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {'error': str(e)}

    def reset_collection(self):
        """Reset/clear the entire collection (use with caution!)"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info(f"Collection '{self.collection_name}' reset successfully")
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise
