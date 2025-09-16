import os
from typing import Dict, List, Optional
from groq import Groq
from src.logger import get_logger
import time
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

class GroqLLMClient:
    """
    Groq LLM client for RAG generation
    Optimized for research/QA applications with proper error handling
    """
    
    def __init__(self, 
                 model_name: str = "openai/gpt-oss-120b", 
                 api_key: Optional[str] = None,
                 max_tokens: int = 3000,
                 temperature: float = 0.1):
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize Groq client
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.client = Groq(api_key=api_key)
        
        # Test connection
        self._test_connection()
        
        logger.info(f"✅ GroqLLM initialized: model={model_name}, max_tokens={max_tokens}")

    def _test_connection(self):
        """Test Groq API connection"""
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model=self.model_name,
                max_tokens=10,
                temperature=0
            )
            logger.info("✅ Groq API connection successful")
        except Exception as e:
            logger.error(f"❌ Groq API connection failed: {e}")
            raise

    def generate_response(self, 
                         prompt: str, 
                         context: str,
                         citations: List[Dict],
                         system_message: Optional[str] = None) -> Dict:
        """
        Generate response using Groq LLM with RAG context
        
        Args:
            prompt: User question
            context: Retrieved context from RAG system
            citations: Citation metadata from retrieval
            system_message: Optional system instruction
            
        Returns:
            Generated response with metadata
        """
        
        start_time = time.time()
        
        # Build system message for research/QA
        if not system_message:
            system_message = self._build_system_message()
        
        # Build user message with context and citations
        user_message = self._build_user_message(prompt, context, citations)
        
        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )
            
            # Extract response
            generated_text = response.choices[0].message.content
            generation_time = time.time() - start_time
            
            # Calculate token usage
            usage = response.usage
            
            result = {
                'response': generated_text,
                'model': self.model_name,
                'generation_time': generation_time,
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
                'citations_used': len(citations),
                'context_length': len(context),
                'finish_reason': response.choices[0].finish_reason
            }
            
            logger.info(f"Generated response: {usage.completion_tokens} tokens in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return {
                'response': f"I apologize, but I encountered an error generating a response: {str(e)}",
                'error': str(e),
                'model': self.model_name,
                'generation_time': time.time() - start_time
            }

    def _build_system_message(self) -> str:
        """Build system message optimized for RAG research/QA"""
        return """You are a knowledgeable research assistant that provides accurate, well-sourced answers based on retrieved documents. Your responses should be:

1. **Accurate**: Only use information from the provided context
2. **Well-cited**: Reference specific sources with [1], [2] notation
3. **Comprehensive**: Provide thorough explanations when possible
4. **Clear**: Use clear, professional language
5. **Honest**: If information is not in the context, say so clearly

When citing sources:
- Use [1], [2], [3] etc. to reference the provided citations
- Place citations immediately after relevant statements
- Include page numbers when available (e.g., "According to page 15[1]...")

If the context doesn't contain sufficient information to answer the question, clearly state what information is missing and what you cannot determine from the provided sources."""

    def _build_user_message(self, prompt: str, context: str, citations: List[Dict]) -> str:
        """Build user message with context and citation information"""
        
        # Build citation reference
        citation_info = "## Available Sources:\n"
        for i, citation in enumerate(citations, 1):
            filename = citation.get('filename', 'Unknown')
            page = citation.get('page_number')
            page_info = f" (Page {page})" if page else ""
            similarity = citation.get('similarity_score', 0)
            
            citation_info += f"[{i}] {filename}{page_info} - Relevance: {similarity:.2f}\n"
        
        # Build complete user message
        user_message = f"""## Question:
{prompt}

{citation_info}

## Retrieved Context:
{context}

## Instructions:
Please answer the question using the retrieved context above. Cite your sources using [1], [2], etc. notation. If the context doesn't fully answer the question, please indicate what information is missing."""
        
        return user_message

    def get_available_models(self) -> List[str]:
        """Get list of available Groq models"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
