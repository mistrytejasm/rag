import os
from src.rag_system import RAGSystem
from src.retrieval_engine import RetrievalEngine
from src.vector_store import VectorStore
from src.embedding_manager import EmbeddingManager
import time
from typing import List, Dict, Any, Annotated

class RerankerABTest:
    """
    A/B testing framework to compare retrieval with and without reranking
    """
    
    def __init__(self):
        # Initialize shared components
        self.vector_store = VectorStore(collection_name="rerank_test")
        self.embedding_manager = EmbeddingManager()
        
        # Create two retrieval engines
        self.retriever_with_rerank = RetrievalEngine(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            enable_reranking=True
        )
        
        self.retriever_without_rerank = RetrievalEngine(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            enable_reranking=False
        )
        
        print("🧪 A/B Test Framework Initialized")
        print("   Group A: Standard Retrieval")
        print("   Group B: Retrieval + Reranking")

    def run_comparison_test(self, test_queries: List[str]) -> Dict:
        """Run A/B comparison test"""
        
        results = {
            'without_reranking': [],
            'with_reranking': [],
            'performance_summary': {}
        }
        
        print(f"\n🚀 Running A/B test with {len(test_queries)} queries")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 Query {i}/{len(test_queries)}: '{query}'")
            
            # Test without reranking
            start_time = time.time()
            results_no_rerank = self.retriever_without_rerank.retrieve_context(
                query, top_k=5, fetch_k=5
            )
            time_no_rerank = time.time() - start_time
            
            # Test with reranking  
            start_time = time.time()
            results_rerank = self.retriever_with_rerank.retrieve_context(
                query, top_k=5, fetch_k=20  # Fetch more for reranking
            )
            time_rerank = time.time() - start_time
            
            # Store results
            results['without_reranking'].append({
                'query': query,
                'chunk_count': results_no_rerank['chunk_count'],
                'avg_similarity': results_no_rerank['avg_similarity'],
                'processing_time': time_no_rerank,
                'citations': results_no_rerank['citations']
            })
            
            results['with_reranking'].append({
                'query': query,
                'chunk_count': results_rerank['chunk_count'],
                'avg_similarity': results_rerank['avg_similarity'],
                'processing_time': time_rerank,
                'citations': results_rerank['citations']
            })
            
            # Print comparison
            print(f"   📊 Without Reranking: {results_no_rerank['chunk_count']} chunks, {results_no_rerank['avg_similarity']:.3f} avg sim, {time_no_rerank:.3f}s")
            print(f"   🎯 With Reranking:    {results_rerank['chunk_count']} chunks, {results_rerank['avg_similarity']:.3f} avg sim, {time_rerank:.3f}s")
            
            if results_rerank['avg_similarity'] > results_no_rerank['avg_similarity']:
                improvement = ((results_rerank['avg_similarity'] - results_no_rerank['avg_similarity']) / results_no_rerank['avg_similarity']) * 100
                print(f"   ✅ Reranking improved similarity by {improvement:.1f}%")
            else:
                print(f"   ❌ Reranking did not improve similarity")
        
        # Calculate summary metrics
        results['performance_summary'] = self._calculate_summary(results)
        
        return results
    
    def _calculate_summary(self, results: Dict) -> Dict:
        """Calculate performance summary metrics"""
        
        no_rerank = results['without_reranking']
        with_rerank = results['with_reranking']
        
        summary = {
            'avg_similarity_no_rerank': sum(r['avg_similarity'] for r in no_rerank) / len(no_rerank),
            'avg_similarity_with_rerank': sum(r['avg_similarity'] for r in with_rerank) / len(with_rerank),
            'avg_time_no_rerank': sum(r['processing_time'] for r in no_rerank) / len(no_rerank),
            'avg_time_with_rerank': sum(r['processing_time'] for r in with_rerank) / len(with_rerank),
            'queries_improved_by_rerank': sum(1 for i in range(len(no_rerank)) 
                                            if with_rerank[i]['avg_similarity'] > no_rerank[i]['avg_similarity']),
            'total_queries': len(no_rerank)
        }
        
        summary['similarity_improvement_pct'] = ((summary['avg_similarity_with_rerank'] - summary['avg_similarity_no_rerank']) / summary['avg_similarity_no_rerank']) * 100
        summary['time_overhead_pct'] = ((summary['avg_time_with_rerank'] - summary['avg_time_no_rerank']) / summary['avg_time_no_rerank']) * 100
        summary['queries_improved_pct'] = (summary['queries_improved_by_rerank'] / summary['total_queries']) * 100
        
        return summary

def test_reranking_comparison():
    """Test reranking vs no reranking"""
    
    # Setup test data
    os.makedirs('sample_docs', exist_ok=True)
    
    # Create detailed test content
    detailed_ml_content = """
    Supervised learning is a machine learning approach where algorithms learn from labeled training data. In supervised learning, you provide the model with input-output pairs, and it learns to map inputs to correct outputs. Common supervised learning algorithms include linear regression, decision trees, random forests, and support vector machines. Examples include email spam detection, image classification, and price prediction.

    Unsupervised learning finds patterns in data without labeled examples. It includes clustering algorithms like K-means that group similar data points, and dimensionality reduction techniques like Principal Component Analysis (PCA) that simplify data while preserving important information. Common applications include customer segmentation and data compression.

    Reinforcement learning learns through trial and error, receiving rewards or penalties for actions. The agent explores an environment and learns optimal strategies to maximize cumulative reward. This approach is used in game playing (like AlphaGo), robotics, and autonomous vehicle navigation.

    Deep learning uses artificial neural networks with multiple layers to model complex patterns in data. These networks can automatically learn hierarchical representations, making them effective for image recognition, natural language processing, and speech recognition. Popular architectures include convolutional neural networks for images and recurrent neural networks for sequences.
    """
    
    ai_detailed_content = """
    Natural Language Processing combines computational linguistics with machine learning to help computers understand human language. Key NLP tasks include sentiment analysis, named entity recognition, machine translation, and text summarization. Modern NLP systems use transformer architectures like BERT and GPT for improved understanding.

    Computer vision enables machines to interpret visual information from images and videos. Core techniques include object detection, image segmentation, and facial recognition. Convolutional neural networks are the foundation of most computer vision systems, with applications in medical imaging, autonomous vehicles, and security systems.

    Artificial Intelligence encompasses the broader goal of creating intelligent machines. AI includes symbolic reasoning, expert systems, and modern machine learning approaches. Current AI systems excel at specific tasks but artificial general intelligence remains a long-term research goal.
    """
    
    with open('sample_docs/detailed_ml.txt', 'w') as f:
        f.write(detailed_ml_content)
    
    with open('sample_docs/detailed_ai.txt', 'w') as f:
        f.write(ai_detailed_content)
    
    # Initialize RAG system and add documents
    rag = RAGSystem(collection_name="rerank_test", chunk_size=200, overlap=40)
    file_paths = ['sample_docs/detailed_ml.txt', 'sample_docs/detailed_ai.txt']
    rag.add_documents(file_paths)
    
    # Initialize A/B test
    ab_test = RerankerABTest()
    
    # Define test queries
    test_queries = [
        "What is supervised learning and how does it work?",
        "Explain the difference between supervised and unsupervised learning",
        "How does reinforcement learning work with rewards?", 
        "What are the applications of computer vision?",
        "What is natural language processing used for?",
        "How do neural networks work in deep learning?",
        "What algorithms are used in supervised learning?",
        "What is the goal of artificial intelligence?"
    ]
    
    # Run A/B comparison
    results = ab_test.run_comparison_test(test_queries)
    
    # Display detailed results
    print("\n" + "="*80)
    print("🏁 A/B TEST RESULTS SUMMARY")
    print("="*80)
    
    summary = results['performance_summary']
    
    print(f"📊 Overall Metrics:")
    print(f"   Average Similarity (No Rerank):  {summary['avg_similarity_no_rerank']:.3f}")
    print(f"   Average Similarity (Reranked):   {summary['avg_similarity_with_rerank']:.3f}")
    print(f"   Similarity Improvement:          {summary['similarity_improvement_pct']:+.1f}%")
    print(f"")
    print(f"⏱️  Performance Metrics:")
    print(f"   Average Time (No Rerank):        {summary['avg_time_no_rerank']:.3f}s")
    print(f"   Average Time (Reranked):         {summary['avg_time_with_rerank']:.3f}s")
    print(f"   Time Overhead:                   {summary['time_overhead_pct']:+.1f}%")
    print(f"")
    print(f"🎯 Quality Metrics:")
    print(f"   Queries Improved by Reranking:   {summary['queries_improved_by_rerank']}/{summary['total_queries']}")
    print(f"   Improvement Rate:                {summary['queries_improved_pct']:.1f}%")
    
    if summary['similarity_improvement_pct'] > 5:
        print(f"\n✅ RECOMMENDATION: Enable reranking - significant improvement detected!")
    elif summary['similarity_improvement_pct'] > 0:
        print(f"\n⚡ RECOMMENDATION: Consider reranking - modest improvement with time overhead")
    else:
        print(f"\n❌ RECOMMENDATION: Skip reranking - no significant benefit for this dataset")
    
    print("="*80)

if __name__ == "__main__":
    test_reranking_comparison()
