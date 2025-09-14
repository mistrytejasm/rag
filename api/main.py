from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from pathlib import Path
import tempfile
import shutil

from src.rag_generator import RAGGenerator
from src.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced RAG System with Groq LLM",
    description="Production-ready RAG system with enhanced retrieval and Groq generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Generator (your complete system)
rag_generator = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_generator
    try:
        rag_generator = RAGGenerator(
            groq_model="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            retrieval_top_k=5,
            enable_reranking=True  # Based on your proven A/B test results
        )
        logger.info("🚀 RAG Generator initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG Generator: {e}")
        raise

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    source_filter: Optional[str] = None
    page_filter: Optional[int] = None
    custom_system_message: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict]
    metadata: Dict

class DocumentUploadResponse(BaseModel):
    message: str
    files_processed: List[str]
    chunks_created: Dict[str, int]
    total_chunks: int

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Advanced RAG System with Groq LLM",
        "status": "operational",
        "version": "1.0.0"
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question using the complete RAG system
    Returns generated answer with sources and citations
    """
    try:
        if not rag_generator:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        logger.info(f"Processing question: {request.question}")
        
        # Generate answer using complete RAG system
        result = rag_generator.generate_answer(
            question=request.question,
            source_filter=request.source_filter,
            page_filter=request.page_filter,
            custom_system_message=request.custom_system_message
        )
        
        return QuestionResponse(
            answer=result['answer'],
            sources=result['sources'],
            metadata=result['metadata']
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...), 
                          background_tasks: BackgroundTasks = None):
    """
    Upload documents to the RAG system
    Supports multiple file formats (PDF, DOCX, TXT, etc.)
    """
    try:
        if not rag_generator:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        # Create temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()
        saved_files = []
        
        try:
            # Save uploaded files
            for file in files:
                if not file.filename:
                    continue
                    
                file_path = Path(temp_dir) / file.filename
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                saved_files.append(str(file_path))
            
            logger.info(f"Saved {len(saved_files)} files for processing")
            
            # Process documents with RAG system
            document_ids = rag_generator.add_documents(saved_files)
            
            # Calculate statistics
            chunks_created = {
                Path(file_path).name: len(ids) 
                for file_path, ids in document_ids.items()
            }
            total_chunks = sum(chunks_created.values())
            
            logger.info(f"Successfully processed {len(saved_files)} files, created {total_chunks} chunks")
            
            return DocumentUploadResponse(
                message=f"Successfully processed {len(saved_files)} documents",
                files_processed=[Path(f).name for f in saved_files],
                chunks_created=chunks_created,
                total_chunks=total_chunks
            )
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        if not rag_generator:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        stats = rag_generator.get_system_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system stats: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        health_status = {
            "status": "healthy",
            "rag_system": "initialized" if rag_generator else "not_initialized",
            "groq_api": "connected" if rag_generator and hasattr(rag_generator.groq_client, 'client') else "not_connected"
        }
        
        # Test Groq connection if possible
        if rag_generator:
            try:
                models = rag_generator.groq_client.get_available_models()
                health_status["groq_models_available"] = len(models) > 0
            except:
                health_status["groq_models_available"] = False
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
