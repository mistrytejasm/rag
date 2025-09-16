from typing import Dict, List, Optional
from src.rag_system import RAGSystem
from src.groq_llm import GroqLLMClient
from src.logger import get_logger

logger = get_logger(__name__)

class RAGGenerator:
    """
    Complete RAG Generation Engine combining your proven retrieval with Groq LLM
    Optimized for research/QA applications with proper source attribution
    """
    
    def __init__(self, 
                 rag_system: Optional[RAGSystem] = None,
                 groq_model: str = "openai/gpt-oss-120b",
                 groq_api_key: Optional[str] = None,
                 retrieval_top_k: int = 5,
                 enable_reranking: bool = True):
        
        # Initialize RAG system (your proven Phase 4 system)
        self.rag_system = rag_system or RAGSystem(
            enable_reranking=enable_reranking,
            chunk_size=1000,
            overlap=200
        )
        
        # Initialize Groq LLM
        self.groq_client = GroqLLMClient(
            model_name=groq_model,
            api_key=groq_api_key,
            max_tokens=3000,
            temperature=0.1
        )
        
        self.retrieval_top_k = retrieval_top_k
        
        logger.info(f"🚀 RAG Generator initialized with reranking: {enable_reranking}")

    def generate_answer(self, 
                       question: str,
                       source_filter: Optional[str] = None,
                       page_filter: Optional[int] = None,
                       custom_system_message: Optional[str] = None) -> Dict:
        """
        Generate complete answer using RAG + Groq LLM
        
        Args:
            question: User question
            source_filter: Optional filter by source file
            page_filter: Optional filter by page number
            custom_system_message: Optional custom system prompt
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        
        logger.info(f"Processing question: '{question}'")
        
        try:
            # Step 1: Retrieve relevant context (your proven system)
            retrieval_results = self.rag_system.query(
                question=question,
                top_k=self.retrieval_top_k,
                source_filter=source_filter,
                page_filter=page_filter
            )
            
            # Check if we have sufficient context
            if retrieval_results['chunk_count'] == 0:
                return {
                    'answer': "I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing your question or check if the relevant documents have been added to the system.",
                    'sources': [],
                    'metadata': {
                        'chunks_retrieved': 0,
                        'avg_similarity': 0.0,
                        'has_context': False,
                        'generation_time': 0,
                        'model_used': self.groq_client.model_name
                    }
                }
            
            # Step 2: Generate response using Groq LLM
            generation_results = self.groq_client.generate_response(
                prompt=question,
                context=retrieval_results['context'],
                citations=retrieval_results['citations'],
                system_message=custom_system_message
            )
            
            # Step 3: Build complete response
            complete_response = {
                'answer': generation_results['response'],
                'sources': self._format_sources(retrieval_results['citations']),
                'metadata': {
                    'chunks_retrieved': retrieval_results['chunk_count'],
                    'avg_similarity': retrieval_results['avg_similarity'],
                    'has_context': True,
                    'generation_time': generation_results.get('generation_time', 0),
                    'model_used': generation_results.get('model', 'unknown'),
                    'prompt_tokens': generation_results.get('prompt_tokens', 0),
                    'completion_tokens': generation_results.get('completion_tokens', 0),
                    'total_tokens': generation_results.get('total_tokens', 0),
                    'retrieval_method': 'enhanced_with_reranking' if hasattr(self.rag_system.retrieval_engine, 'enable_reranking') else 'standard',
                    'finish_reason': generation_results.get('finish_reason', 'unknown')
                }
            }
            
            logger.info(f"Generated answer: {complete_response['metadata']['completion_tokens']} tokens, {complete_response['metadata']['chunks_retrieved']} sources")
            
            return complete_response
            
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'metadata': {
                    'error': str(e),
                    'has_context': False,
                    'model_used': self.groq_client.model_name
                }
            }

    def _format_sources(self, citations: List[Dict]) -> List[Dict]:
        """Format citations for user-friendly display"""
        formatted_sources = []
        
        for citation in citations:
            source = {
                'id': citation.get('id'),
                'filename': citation.get('filename', 'Unknown'),
                'similarity_score': citation.get('similarity_score', 0),
                'preview': citation.get('content_preview', '')[:200] + "..."
            }
            
            # Add page information if available
            if citation.get('page_number'):
                source['page'] = citation['page_number']
                
            # Add file path for reference
            if citation.get('source_file'):
                source['file_path'] = citation['source_file']
            
            formatted_sources.append(source)
        
        return formatted_sources

    def add_documents(self, file_paths: List[str]) -> Dict:
        """Add documents to the RAG system"""
        return self.rag_system.add_documents(file_paths)

    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        rag_stats = self.rag_system.get_system_stats()
        groq_models = self.groq_client.get_available_models()
        
        return {
            'rag_system': rag_stats,
            'llm': {
                'model': self.groq_client.model_name,
                'available_models': groq_models,
                'max_tokens': self.groq_client.max_tokens,
                'temperature': self.groq_client.temperature
            },
            'configuration': {
                'retrieval_top_k': self.retrieval_top_k,
                'reranking_enabled': hasattr(self.rag_system.retrieval_engine, 'enable_reranking')
            }
        }

    def chat_with_history(self, 
                         question: str, 
                         conversation_history: List[Dict] = None,
                         **kwargs) -> Dict:
        """
        Enhanced chat with conversation context (future enhancement)
        Currently returns single-turn response
        """
        # For now, process as single question
        # TODO: Implement conversation memory in future version
        return self.generate_answer(question, **kwargs)
