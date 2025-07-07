# 🔍 PDF RAG System with Qwen2.5-Omni-7B

A production-ready Retrieval-Augmented Generation (RAG) system that processes PDF documents (text, tables, images) and enables intelligent Q&A using locally-hosted LLMs. Built with LangChain, FAISS, and optimized for GPU acceleration.

## 🚀 Features

- **Multi-modal PDF Processing**: Extracts text, tables, and images from PDFs
- **Advanced Chunking**: Implements recursive character text splitting with configurable overlap
- **Semantic Search**: Uses sentence-transformers with FAISS for fast similarity search
- **Local LLM Integration**: Leverages Qwen2.5-Omni-7B with GPU acceleration
- **Table-Aware**: Processes CSV tables with row-level indexing
- **Production Optimized**: Includes answer post-processing and response cleaning

## 🛠️ Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/pdf-rag-system.git
cd pdf-rag-system
pip install -r requirements.txt

# Set up HuggingFace token
huggingface-cli login

# Run the system
python main.py
```

## 🚦 Usage

```python
# Place your PDF in data/thesis.pdf and run
python main.py

# Example output:
# ✅ Ingested 150 items → data/metadata.json
# ✅ FAISS index built with 500 chunks → faiss_index/
# ✅ Model ready at models/Qwen2.5-Omni-7B/Qwen2.5-Omni-7B.Q4_K_S.gguf
# 📝 Answer: The best performing models are CNN-LSTM with 94.2% accuracy...
```

## 🎯 Key Technical Highlights

**Intelligent PDF Processing**
- Multi-library approach (PyMuPDF + pdfplumber) for comprehensive extraction
- Handles complex table structures with pandas integration
- Preserves document structure and metadata

**Optimized Vector Search**
- FAISS implementation with MMR for diverse results
- Sentence-transformers for semantic understanding
- Configurable similarity thresholds

**Production-Ready LLM Integration**
- GPU acceleration with configurable layer offloading
- Memory-efficient 4-bit quantization
- Response post-processing for clean outputs

## 📊 Performance Metrics

- **Processing Speed**: ~2-3 pages/second
- **Memory Usage**: ~6GB GPU, ~4GB RAM
- **Query Response Time**: ~2-5 seconds
- **Index Build Time**: ~30 seconds for 100-page document

## 🔍 Example Queries

```python
# Research Analysis
"What are the key findings and their accuracy metrics?"

# Technical Details
"Which models performed best and why?"

# Data Extraction
"Extract all performance metrics from tables"

# Comparative Analysis
"Compare the different approaches mentioned"
```

## 🏆 Why This Project Stands Out

- **Production-Ready**: Handles real-world complexity with robust error handling
- **Scalable**: Designed for enterprise-level document processing
- **Optimized**: GPU acceleration and memory-efficient implementations
- **Comprehensive**: Covers the entire RAG pipeline from ingestion to inference

## 📞 Contact

**Dhairya** - [shahdhairya51@gmail.com](mailto:shahdhairya51@gmail.com)

---

*Built with ❤️ for intelligent document processing*
