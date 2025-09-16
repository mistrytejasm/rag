import streamlit as st
import tempfile
import os
from pathlib import Path
import time
from typing import List, Dict
import pandas as pd
import json
import re
from bs4 import BeautifulSoup
import io

from src.rag_generator import RAGGenerator
from src.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better UI
st.markdown("""
<style>
    /* Global styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        width: 300px !important;
        padding: 0.5rem 1rem 0.5rem 1rem !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-radius: 10px;
    }
    
    /* Content rendering */
    .stMarkdown p {
        font-size: 16px;
        line-height: 1.6;
        color: #262730;
        margin-bottom: 1rem;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1f77b4;
        margin-top: 0.5rem !important;
        margin-bottom: 1rem;
    }
            
    .stMarkdown h2 {
        font-size: 1.5rem;
        margin-top: 1rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    
    /* Header styling */
    .main-header {
        font-size: 1.5rem !important;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
        background: linear-gradient(90deg, #1f77b4, #2196f3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* File type badges */
    .file-type-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Metrics styling */
    .metric-row {
        display: flex;
        gap: 1rem;
        justify-content: space-between;
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Document content containers */
    .document-content {
        background-color: #fafafa;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        font-family: 'Georgia', serif;
        line-height: 1.7;
    }
    
    /* Table styling */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* JSON display */
    .stJson {
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        font-family: 'Courier New', monospace;
    }
    
    # /* Sidebar sections */
    # .sidebar-section {
    #     background: white;
    #     padding: 1rem;
    #     border-radius: 10px;
    #     margin-bottom: 1.5rem;
    #     box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    # }
    
    .sidebar-header {
        color: #1f77b4;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e3f2fd;
    }
    
    /* File list styling */
    .file-item {
        display: flex;
        align-items: center;
        padding: 0.6rem;
        margin: 0.3rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 8px;
        font-size: 14px;
        border-left: 3px solid #1f77b4;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Remove scrolling issues */
    .main, .block-container {
        overflow-x: hidden;
    }
    
    /* Source preview styling */
    .source-preview {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

def detect_content_type(filename: str) -> str:
    """Detect document type from filename"""
    extension = Path(filename).suffix.lower()
    type_mapping = {
        '.txt': 'text',
        '.md': 'markdown', 
        '.pdf': 'pdf',
        '.docx': 'document',
        '.csv': 'table',
        '.xlsx': 'table',
        '.html': 'html',
        '.json': 'json'
    }
    return type_mapping.get(extension, 'text')

def get_file_icon(content_type: str) -> str:
    """Get icon for file type"""
    icons = {
        'pdf': '📕',
        'document': '📘', 
        'table': '📊',
        'json': '🔧',
        'html': '🌐',
        'markdown': '📝',
        'text': '📄'
    }
    return icons.get(content_type, '📄')

def clean_and_format_content(content: str, content_type: str = 'text') -> str:
    """Clean and properly format content based on type"""
    if not content:
        return ""
    
    if content_type == 'html':
        # Clean HTML content
        soup = BeautifulSoup(content, 'html.parser')
        # Replace <br> tags with newlines
        for br in soup.find_all(['br', 'BR']):
            br.replace_with('\n')
        content = soup.get_text()
    
    # Replace HTML break tags with proper line breaks for all types
    content = re.sub(r'<br\s*/?>', '\n', content, flags=re.IGNORECASE)
    content = re.sub(r'<BR\s*/?>', '\n', content, flags=re.IGNORECASE)
    
    # Clean up extra whitespace but preserve intentional formatting
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip()
    
    return content

def render_content_by_type(content: str, filename: str, metadata: dict = None):
    """Render content based on document type"""
    content_type = detect_content_type(filename)
    icon = get_file_icon(content_type)
    
    # Show file info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{icon} {filename}**")
    with col2:
        st.markdown(f'<span class="file-type-badge">{content_type}</span>', unsafe_allow_html=True)
    
    if content_type == 'table':
        # Handle CSV/Excel files
        try:
            if filename.lower().endswith(('.csv')):
                df = pd.read_csv(io.StringIO(content))
            elif filename.lower().endswith(('.xlsx', '.xls')):
                # For Excel, content is already converted to CSV format by backend
                df = pd.read_csv(io.StringIO(content))
            else:
                st.text(content)
                return
            
            # Display table with pagination for large datasets
            if len(df) > 50:
                st.info(f"📊 Large dataset detected ({len(df)} rows, {len(df.columns)} columns)")
                
                # Show first 50 rows by default
                st.dataframe(df.head(50), use_container_width=True)
                
                # Option to view full table
                with st.expander(f"📋 View Complete Table ({len(df)} rows)", expanded=False):
                    st.dataframe(df, use_container_width=True, height=600)
                
                # Download option
                csv_download = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_download,
                    file_name=f"{Path(filename).stem}_export.csv",
                    mime="text/csv"
                )
            else:
                st.dataframe(df, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Could not parse as table: {e}")
            st.text_area("Raw Content", content, height=200, disabled=True)
    
    elif content_type == 'json':
        # Handle JSON files
        try:
            json_data = json.loads(content)
            st.json(json_data)
        except Exception as e:
            st.warning(f"Could not parse JSON: {e}")
            st.code(content, language='json')
    
    elif content_type in ['pdf', 'document']:
        # Handle PDF/DOCX with potential page information
        cleaned_content = clean_and_format_content(content, content_type)
        
        if metadata and metadata.get('has_page_info') and metadata.get('total_pages', 0) > 1:
            st.info(f"📑 Document has {metadata['total_pages']} pages")
            
            # Show content in a scrollable container
            st.markdown(f'<div class="document-content">{cleaned_content}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(cleaned_content)
    
    elif content_type == 'html':
        # Handle HTML files
        cleaned_content = clean_and_format_content(content, content_type)
        st.markdown(cleaned_content)
    
    else:
        # Handle text/markdown and other formats
        cleaned_content = clean_and_format_content(content, content_type)
        
        if content_type == 'markdown':
            st.markdown(cleaned_content)
        else:
            # For plain text, use a styled container
            st.text_area(
                "Document Content",
                value=cleaned_content,
                height=300,
                disabled=True,
                label_visibility="collapsed"
            )

def display_enhanced_sources(sources: list):
    """Display sources with enhanced formatting and type awareness"""
    if not sources:
        return
        
    with st.expander(f"📚 Source References ({len(sources)} sources)", expanded=False):
        for i, source in enumerate(sources, 1):
            filename = source.get('filename', 'Unknown')
            content_type = detect_content_type(filename)
            icon = get_file_icon(content_type)
            
            # Source header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{icon} [{i}] {filename}**")
                st.caption(f"Page: {source.get('page', 'N/A')} | Similarity: {source['similarity_score']:.3f}")
            with col2:
                st.markdown(f'<span class="file-type-badge">{content_type}</span>', 
                           unsafe_allow_html=True)
            
            # Content preview
            preview_content = source.get('preview', '')
            if preview_content:
                with st.expander(f"Preview Content", expanded=False):
                    if content_type == 'table':
                        # For tables, show first few lines
                        lines = preview_content.split('\n')[:5]
                        st.code('\n'.join(lines) + '\n... (truncated)')
                    elif content_type == 'json':
                        try:
                            json_preview = json.loads(preview_content[:500])
                            st.json(json_preview)
                        except:
                            st.code(preview_content[:300] + "...")
                    else:
                        cleaned_preview = clean_and_format_content(preview_content, content_type)
                        display_text = cleaned_preview[:400] + "..." if len(cleaned_preview) > 400 else cleaned_preview
                        st.markdown(f'<div class="source-preview">{display_text}</div>', 
                                   unsafe_allow_html=True)
            
            if i < len(sources):
                st.divider()

def initialize_rag_system():
    """Initialize RAG system with session state caching"""
    if 'rag_generator' not in st.session_state:
        try:
            with st.spinner("🚀 Initializing Advanced RAG System..."):
                st.session_state.rag_generator = RAGGenerator(
                    groq_model="openai/gpt-oss-120b",
                    retrieval_top_k=10,
                    enable_reranking=True
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
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
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
        file_paths = []
        for file in files:
            saved_path = save_uploaded_file(file)
            if saved_path:
                file_paths.append(saved_path)
        
        if not file_paths:
            return {"success": False, "message": "Failed to save uploaded files"}
        
        with st.spinner(f"Processing {len(file_paths)} documents..."):
            result = st.session_state.rag_generator.add_documents(file_paths)
        
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

def render_sidebar():
    """Render enhanced sidebar with organized sections"""
    with st.sidebar:
        # Document Management Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-header">📁 Document Management</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt', 'md', 'csv', 'xlsx', 'html', 'json'],
            accept_multiple_files=True,
            help="📎 Supported formats: PDF, DOCX, TXT, MD, CSV, XLSX, HTML, JSON\n📏 Max file size: 200MB per file",
            key="file_uploader"
        )
        
        if st.button("🔄 Process Documents", type="primary", use_container_width=True):
            if uploaded_files:
                result = process_uploaded_files(uploaded_files)
                
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                    st.info(f"📊 Created {result['total_chunks']} searchable chunks")
                    
                    # Add file info with types
                    for file in uploaded_files:
                        file_info = {
                            'name': file.name,
                            'type': detect_content_type(file.name),
                            'size': file.size
                        }
                        st.session_state.uploaded_files.append(file_info)
                    
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")
            else:
                st.warning("⚠️ Please upload files first!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Uploaded Files Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-header">📋 Uploaded Files</div>', unsafe_allow_html=True)
        
        if st.session_state.uploaded_files:
            # Group files by type
            file_types = {}
            for file_info in st.session_state.uploaded_files:
                if isinstance(file_info, dict):
                    file_type = file_info.get('type', 'text')
                    filename = file_info.get('name')
                    size = file_info.get('size', 0)
                else:
                    file_type = detect_content_type(file_info)
                    filename = file_info
                    size = 0
                
                if file_type not in file_types:
                    file_types[file_type] = []
                file_types[file_type].append({'name': filename, 'size': size})
            
            # Display files grouped by type
            for file_type, files in file_types.items():
                icon = get_file_icon(file_type)
                with st.expander(f"{icon} {file_type.title()} ({len(files)})", expanded=True):
                    for file_data in files:
                        filename = file_data['name']
                        size_mb = file_data['size'] / (1024*1024) if file_data['size'] > 0 else 0
                        size_str = f"({size_mb:.1f} MB)" if size_mb > 0 else ""
                        st.markdown(f'<div class="file-item">✅ {filename} {size_str}</div>', 
                                   unsafe_allow_html=True)
        else:
            st.info("📝 No files uploaded yet")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System Statistics Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-header">📊 System Statistics</div>', unsafe_allow_html=True)
        
        try:
            stats = st.session_state.rag_generator.get_system_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📚 Documents", stats['rag_system']['vector_store']['total_documents'])
            with col2:
                reranking_status = "ON" if stats['configuration']['reranking_enabled'] else "OFF"
                st.metric("🔄 Reranking", reranking_status)
            
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin-top: 1rem;">
                <strong>🤖 Model:</strong> {stats['llm']['model']}<br>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading stats: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Actions Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-header">⚙️ Actions</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_chat_message(message: Dict, key: str):
    """Display chat message with enhanced formatting"""
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            # Clean and format the content properly
            formatted_content = clean_and_format_content(content)
            st.markdown(formatted_content)
            
            # Display metadata if available
            if "metadata" in message and message["metadata"]:
                metadata = message["metadata"]
                
                with st.expander("📊 Response Analytics", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            label="📦 Chunks Retrieved",
                            value=metadata.get('chunks_retrieved', 0),
                            help="Number of document chunks used"
                        )
                    with col2:
                        st.metric(
                            label="🎯 Avg Similarity",
                            value=f"{metadata.get('avg_similarity', 0):.3f}",
                            help="Average similarity score"
                        )
                    with col3:
                        st.metric(
                            label="⏱️ Generation Time",
                            value=f"{metadata.get('generation_time', 0):.2f}s",
                            help="Time taken to generate response"
                        )
            
            # Display sources with enhanced formatting
            if "sources" in message and message["sources"]:
                display_enhanced_sources(message["sources"])

def main():
    """Main application function"""
    
    # Initialize everything
    initialize_session_state()
    initialize_rag_system()
    
    # Main header with enhanced styling
    # st.markdown('<h1 class="main-header">Advanced RAG Assistant</h1>', unsafe_allow_html=True)    
    # Render sidebar
    render_sidebar()
    
    # Main chat interface
    st.markdown("## 💬 Chat with Your Documents")
    
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
        
        # Display user message immediately
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Analyzing documents and generating response..."):
                try:
                    result = st.session_state.rag_generator.generate_answer(prompt)
                    
                    # Clean and display response
                    formatted_response = clean_and_format_content(result['answer'])
                    st.markdown(formatted_response)
                    
                    # Display metadata
                    if result['metadata'].get('chunks_retrieved', 0) > 0:
                        with st.expander("📊 Response Analytics", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📦 Chunks Retrieved", result['metadata']['chunks_retrieved'])
                            with col2:
                                st.metric("🎯 Avg Similarity", f"{result['metadata']['avg_similarity']:.3f}")
                            with col3:
                                st.metric("⏱️ Generation Time", f"{result['metadata']['generation_time']:.2f}s")
                    
                    # Display sources
                    if result['sources']:
                        display_enhanced_sources(result['sources'])
                    
                    # Add assistant message to chat history
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
