import streamlit as st
import tempfile
import os
from pathlib import Path
import time
from typing import List, Dict
import pandas as pd

# Import your existing RAG system
from src.rag_generator import RAGGenerator
from src.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="🧠 Advanced RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    
    .stats-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_rag_system():
    """Initialize RAG system with session state caching"""
    if 'rag_generator' not in st.session_state:
        try:
            with st.spinner("🚀 Initializing Advanced RAG System..."):
                st.session_state.rag_generator = RAGGenerator(
                    groq_model="openai/gpt-oss-120b",  # Your preferred model
                    retrieval_top_k=5,
                    enable_reranking=True  # Based on your proven +2,352% improvement
                )
            st.success("✅ RAG System initialized successfully!")
            logger.info("RAG System initialized in Streamlit")
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {str(e)}")
            st.stop()

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory"""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file {uploaded_file.name}: {str(e)}")
        return None

def process_uploaded_files(files: List) -> Dict:
    """Process uploaded files using RAG system"""
    if not files:
        return {"success": False, "message": "No files to process"}
    
    try:
        # Save files to temporary location
        file_paths = []
        for file in files:
            saved_path = save_uploaded_file(file)
            if saved_path:
                file_paths.append(saved_path)
        
        if not file_paths:
            return {"success": False, "message": "Failed to save uploaded files"}
        
        # Process with RAG system
        with st.spinner(f"🔄 Processing {len(file_paths)} documents..."):
            result = st.session_state.rag_generator.add_documents(file_paths)
        
        # Calculate statistics
        total_chunks = sum(len(ids) for ids in result.values())
        processed_files = len([ids for ids in result.values() if len(ids) > 0])
        
        # Clean up temporary files
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except:
                pass
        
        return {
            "success": True,
            "message": f"Successfully processed {processed_files} documents",
            "total_chunks": total_chunks,
            "processed_files": processed_files,
            "details": result
        }
        
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return {"success": False, "message": f"Error processing files: {str(e)}"}

def display_chat_message(message: Dict, key: str):
    """Display a chat message with proper styling"""
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    else:
        with st.chat_message("assistant"):
            st.write(content)
            
            # Display metadata if available
            if "metadata" in message and message["metadata"]:
                metadata = message["metadata"]
                
                with st.expander("📊 Response Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Chunks Retrieved", metadata.get('chunks_retrieved', 0))
                    with col2:
                        st.metric("Avg Similarity", f"{metadata.get('avg_similarity', 0):.3f}")
                    with col3:
                        st.metric("Generation Time", f"{metadata.get('generation_time', 0):.2f}s")
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**[{i}] {source['filename']}**")
                        st.write(f"Similarity: {source['similarity_score']:.3f}")
                        if source.get('page'):
                            st.write(f"Page: {source['page']}")
                        st.write(f"Preview: {source['preview'][:100]}...")
                        st.divider()

def main():
    """Main application function"""
    
    # Initialize everything
    initialize_session_state()
    initialize_rag_system()
    
    # Main header
    st.markdown('<h1 class="main-header">🧠 Advanced RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by Enhanced Retrieval (+2,352% improvement) & Groq LLM**")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Support for PDF, DOCX, and TXT files"
        )
        
        # Process button
        if st.button("🔄 Process Documents", type="primary"):
            if uploaded_files:
                result = process_uploaded_files(uploaded_files)
                
                if result["success"]:
                    st.success(result["message"])
                    st.info(f"📊 Created {result['total_chunks']} searchable chunks")
                    
                    # Update session state
                    st.session_state.uploaded_files.extend([f.name for f in uploaded_files])
                    
                    # Rerun to update the interface
                    st.rerun()
                else:
                    st.error(result["message"])
            else:
                st.warning("Please upload files first!")
        
        # Display uploaded files
        st.divider()
        st.subheader("📋 Uploaded Files")
        if st.session_state.uploaded_files:
            for file_name in st.session_state.uploaded_files:
                st.write(f"✅ {file_name}")
        else:
            st.write("*No files uploaded yet*")
        
        # System statistics
        st.divider()
        st.subheader("📊 System Stats")
        
        try:
            stats = st.session_state.rag_generator.get_system_stats()
            
            st.metric(
                "Total Documents", 
                stats['rag_system']['vector_store']['total_documents']
            )
            st.metric(
                "LLM Model", 
                stats['llm']['model']
            )
            st.metric(
                "Reranking", 
                "✅ Enabled" if stats['configuration']['reranking_enabled'] else "❌ Disabled"
            )
            
        except Exception as e:
            st.error(f"Error loading stats: {e}")
        
        # Clear conversation button
        st.divider()
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat with Your Documents")
    
    # Display existing messages
    for i, message in enumerate(st.session_state.messages):
        display_chat_message(message, key=f"message_{i}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your uploaded documents..."):
        
        # Check if documents are uploaded
        if not st.session_state.uploaded_files:
            st.warning("⚠️ Please upload and process some documents first!")
            return
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Generate response using your RAG system
                    result = st.session_state.rag_generator.generate_answer(prompt)
                    
                    # Display response
                    st.write(result['answer'])
                    
                    # Display metadata
                    if result['metadata'].get('chunks_retrieved', 0) > 0:
                        with st.expander("📊 Response Details"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Chunks Retrieved", result['metadata']['chunks_retrieved'])
                            with col2:
                                st.metric("Avg Similarity", f"{result['metadata']['avg_similarity']:.3f}")
                            with col3:
                                st.metric("Generation Time", f"{result['metadata']['generation_time']:.2f}s")
                    
                    # Display sources
                    if result['sources']:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(result['sources'], 1):
                                st.write(f"**[{i}] {source['filename']}**")
                                st.write(f"Similarity: {source['similarity_score']:.3f}")
                                if source.get('page'):
                                    st.write(f"Page: {source['page']}")
                                st.write(f"Preview: {source['preview'][:100]}...")
                                if i < len(result['sources']):
                                    st.divider()
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "metadata": result['metadata'],
                        "sources": result['sources']
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()
