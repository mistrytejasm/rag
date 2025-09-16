# 🧠 Advanced RAG Assistant

Retrieval-Augmented Generation (RAG) system for advanced document understanding and question answering. It supports multiple document formats, enhanced retrieval with cross-encoder reranking, and a professional Streamlit UI. 

---

## 🌟 Key Features

- 🔍 Enhanced retrieval with cross-encoder reranking (higher-precision context selection)
- 📚 Multi-format document support: PDF, DOCX, TXT, MD, CSV, XLSX, HTML, JSON
- 🤖 Groq LLM integration with configurable parameters
- 🎨 Modern Streamlit UI with source previews and response analytics
- ⚙️ Modular Python backend (clean separation of concerns)
- 🧾 Comprehensive logging and error handling

---

## 📁 Project Structure

```
RAG/
├── docs/                         # Documentation & samples (keep non-sensitive docs)
├── logs/                         # Runtime logs (git-ignored recommended)
├── src/                          # Core application modules
│   ├── chunker.py                # Token-aware/semantic chunking
│   ├── config.py                 # Central configuration access
│   ├── embedding_manager.py      # Embedding model wrapper + caching hooks
│   ├── embedding_pipeline.py     # Batch embedding pipeline
│   ├── file_processor.py         # Multi-format ingestion (PDF/DOCX/CSV/HTML/JSON/etc.)
│   ├── groq_llm.py               # Groq LLM client wrapper
│   ├── logger.py                 # Structured logging
│   ├── pipeline_manager.py       # Orchestration of processing stages
│   ├── rag_generator.py          # Query -> retrieve -> generate pipeline
│   ├── rag_system.py             # System composition utilities
│   ├── reranker.py               # Cross-encoder reranking
│   ├── retrieval_engine.py       # Vector + hybrid retrieval
│   └── vector_store.py           # Vector DB abstraction (Chroma/FAISS adapters)
├── temp_uploads/                 # Ephemeral uploads (cleaned after processing)
├── tests/                        # Tests and experiments
│   ├── ab_test_reranking.py      # A/B tests for reranker
│   ├── test_complete_system.py   # End-to-end tests
│   └── Test_Files.ipynb          # Optional interactive test notebook
├── vector_db/                    # Local vector DB storage (git-ignored recommended)
├── .env                          # Environment variables (never commit secrets)
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
└── streamlit_app.py              # Streamlit frontend application (single entrypoint)
```
---

## 🧰 Prerequisites

- Python 3.8+
- A valid Groq API key (and any other keys you intend to use)

---

## 🚀 Installation

```
# 1) Clone the repository
git clone https://github.com/mistrytejasm/rag.git
cd advanced-rag-assistant

# 2) Create and activate a virtual environment
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Configure environment variables
# create .env and set:
# GROQ_API_KEY=your_api_key_here
# RAG_SIMILARITY_THRESHOLD=0.0
# RAG_CHUNK_SIZE=500
# RAG_CHUNK_OVERLAP=100
# RAG_RETRIEVAL_TOP_K=5
# RAG_ENABLE_RERANKING=true
```

### 📄 Example `.env`

```
# LLM
GROQ_API_KEY=your_groq_api_key_here

# Retrieval & chunking
RAG_SIMILARITY_THRESHOLD=0.0
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=100
RAG_RETRIEVAL_TOP_K=5
RAG_ENABLE_RERANKING=true

# Optional toggles
RAG_ENABLE_PREPROCESSING=true
RAG_ENABLE_CACHE=true
```

---

## 💻 Running the App (Streamlit)

```
streamlit run streamlit_app.py
```

- The app opens at `http://localhost:8501`
- Use the sidebar to upload documents (PDF, DOCX, TXT, MD, CSV, XLSX, HTML, JSON)
- Click “Process Documents” to index
- Ask natural-language questions in the chat
- Expand “📚 Source References” and “📊 Response Analytics” for details

---

## 🎯 Usage Tips

- For PDFs/DOCX: ensure readable text (scanned images require OCR outside this scope)
- For CSV/XLSX: large tables are previewed with truncation; full data available via expander
- For HTML/JSON: content is cleaned and pretty-printed for readability
- Use specific, concise queries; enable preprocessing in `.env` for synonym expansion

---

## ⚙️ Configuration & Tuning

Adjust parameters in `src/config.py` and `.env`:

| Parameter                  | Description                           | Default |
|---------------------------|---------------------------------------|---------|
| `RAG_SIMILARITY_THRESHOLD`| Minimum similarity for retrieval       | 0.0     |
| `RAG_CHUNK_SIZE`          | Tokens/characters per chunk            | 500     |
| `RAG_CHUNK_OVERLAP`       | Overlap between chunks                 | 100     |
| `RAG_RETRIEVAL_TOP_K`     | Number of chunks to retrieve           | 5       |
| `RAG_ENABLE_RERANKING`    | Enable cross-encoder reranking         | true    |
| `RAG_ENABLE_PREPROCESSING`| Query cleaning/expansion               | true    |
| `RAG_ENABLE_CACHE`        | In-memory caching for speed            | true    |

---

## 📊 Performance & Observability

- Enable reranking for best precision
- Tune chunk size/overlap by document type
- Add caching for repeated queries/embeddings
- Review logs in `logs/` for debugging

---

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch: `git checkout -b feature-name`  
3. Commit: `git commit -m "Add feature"`  
4. Push: `git push origin feature-name`  
5. Open a Pull Request

Please include tests for new features and update documentation as needed.

---

## 📜 License

MIT License. See `LICENSE` for full text.

---

## 🙏 Acknowledgments

- Groq for high-performance LLMs
- Sentence-Transformers for embeddings
- ChromaDB / FAISS for vector search
- Streamlit for the frontend

---

## 📞 Contact

- Email: mistrytejasm@gmail.com.com  

---

**⭐ If this project helps, please star the repository!**  
*Built with ❤️ for the AI community.*
```
