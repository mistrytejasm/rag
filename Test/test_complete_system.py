import os
import asyncio
from src.rag_generator import RAGGenerator
from src.logger import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

async def test_complete_rag_system():
    """Test the complete RAG system with Groq LLM"""
    
    print("🧪 Testing Complete RAG System with Groq LLM Integration")
    
    # Check environment
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Please set GROQ_API_KEY environment variable")
        return
    
    try:
        # Initialize complete RAG system
        rag_generator = RAGGenerator(
            groq_model="openai/gpt-oss-120b",
            retrieval_top_k=3,
            enable_reranking=True  # Based on your proven A/B results
        )
        
        print("✅ RAG Generator initialized successfully")
        
        # Test with existing documents (from your previous tests)
        test_questions = [
            "What is machine learning and how does it work?",
            "Explain the difference between supervised and unsupervised learning",
            "What are the applications of deep learning in AI?",
            "How does natural language processing help computers understand human language?"
        ]
        
        print(f"\n🔍 Testing {len(test_questions)} questions:")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📋 Question {i}: {question}")
            
            # Generate complete answer
            result = rag_generator.generate_answer(question)
            
            # Display results
            print(f"📊 Metadata:")
            print(f"   Chunks retrieved: {result['metadata']['chunks_retrieved']}")
            print(f"   Avg similarity: {result['metadata']['avg_similarity']:.3f}")
            print(f"   Generation time: {result['metadata']['generation_time']:.2f}s")
            print(f"   Total tokens: {result['metadata'].get('total_tokens', 'N/A')}")
            
            print(f"🎯 Answer: {result['answer'][:200]}...")
            
            print(f"📖 Sources:")
            for j, source in enumerate(result['sources'][:2], 1):
                print(f"   [{j}] {source['filename']} - Similarity: {source['similarity_score']:.3f}")
        
        # Test system stats
        print(f"\n📊 System Statistics:")
        stats = rag_generator.get_system_stats()
        print(f"   Total documents: {stats['rag_system']['vector_store']['total_documents']}")
        print(f"   LLM model: {stats['llm']['model']}")
        print(f"   Reranking enabled: {stats['configuration']['reranking_enabled']}")
        
        print(f"\n🎉 Complete RAG system test successful!")
        
    except Exception as e:
        logger.error(f"❌ Complete system test failed: {e}")
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_complete_rag_system())
