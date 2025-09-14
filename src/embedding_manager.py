import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import time
from src.logger import get_logger

logger = get_logger(__name__)

class EmbeddingManager:
  """
	Simple, focused embedding manager using all-mpnet-base-v2
	Perfect for RAG applications with excellent semantic understanding
	"""
  def __init__(self):
    self.model_name = "all-mpnet-base-v2"
    self.embedding_dim = 768   # all-mpnet-base-v2 dimension
    self.model = None
    self._load_model()
    logger.info(f"EmbeddingManager initialized with {self.model_name}")


  def _load_model(self):
    """Load the all-mpnet-base-v2-model"""
    try:
      logger.info("Loding all-mpnet-base-v2-model...")
      self.model = SentenceTransformer(self.model_name)
      logger.info(f"Model Loaded Successfully, embedding dimention: {self.embedding_dim}")

    except Exception as e:
      logger.info(f"Failed tp Load Model: {e}")
      raise

  def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for list of texts using all-mpnet-base-v2
    """

    if not texts:
      return np.array([])
    
    logger.info(f"Generating  embeddings for len{len(texts)} texts")
    start_time = time.time()

    try:
      # generate embeddings in batches foe memory  efficiency
      embeddings = self.model.encode(
        texts,
        batch_size=batch_size,
        show_progress_size=len(texts) > 50, # show progress bar for large batches
        convert_to_tensor=False,
        normalize_embeddings=True, # Normalize for better similarity computation
      )

      end_time = time.time()
      processing_speed = len(texts) / (start_time - end_time)

      logger.info(f"Generated {len(embeddings)} embeddings in {end_time - start_time:.2f}s ({processing_speed:.1f}) texts/sec")

      return np.array(embeddings)
    
    except Exception as e:
      logger.error(f"Failed to Generate Embeddings: {e}")
      raise

  def compute_simiarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings"""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
  
  def get_model_info(self) -> Dict:
    """Get Model Information"""
    return {
      "model_name": self.model_name,
      "embedding_dimention": self.embedding_dim,
      "model_type": 'sentence-transformer',
      "is_loaded": self.model is not None
    }
  

