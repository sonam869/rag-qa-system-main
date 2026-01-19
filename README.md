# Vector Database Setup - Quick Start Guide

This project sets up a hosted vector database using **Pinecone** for storing and querying embeddings.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# - PINECONE_API_KEY: Get from https://app.pinecone.io
# - OPENAI_API_KEY: (Optional) Get from https://platform.openai.com
```

### 3. Initialize Vector Database
```bash
python vector_db_setup.py
```

This will create a Pinecone index with the following configuration:
- **Index Name**: `tumpa-embeddings`
- **Dimension**: 384 (FastEmbed compatible)
- **Metric**: cosine
- **Cloud**: AWS us-east-1 (Serverless)

### 4. Run Examples
```bash
python example_usage.py
```

### 5. Launch Streamlit Frontend
```bash
streamlit run app.py
```
This will open a web interface where you can:
- Upload or paste documents
- Ask questions
- View answers with citations and sources

## Project Structure

```
tumpa/
├── app.py                      # Streamlit frontend
├── vector_db_setup.py          # Main setup script
├── example_usage.py            # Usage examples
├── embeddings.py               # FastEmbed local embeddings
├── chunking.py                 # Text chunking with metadata
├── retriever.py                # MMR retrieval
├── reranker.py                 # FlashRank reranking
├── llm_answering.py            # Groq answer generation
├── rag_pipeline.py             # Complete RAG pipeline
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── VECTOR_DB_DOCUMENTATION.md # Vector DB details
└── RAG_CONFIGURATION.md       # Complete RAG setup guide
```

## Key Configuration

| Property | Value |
|----------|-------|
| Index Name | `tumpa-embeddings` |
| Dimension | 384 (FastEmbed) |
| Metric | cosine |
| Deployment | Serverless (AWS us-east-1) |
| Chunking | 800-1200 tokens, 10-15% overlap |
| Retrieval | MMR with diversity |
| Reranker | FlashRank (free, local) |
| LLM | Groq Llama-3.1-8b |

## Upsert Strategy

- **Batch size**: 100 vectors per batch
- **Operation**: Upsert (update or insert)
- **ID format**: `{source}_{identifier}_{chunk}`
- **Metadata**: Stores original text and custom fields

## Usage Example

### Complete RAG Pipeline
```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Ingest documents
documents = [
    {
        "text": "Your document content...",
        "source": "doc.pdf",
        "title": "Document Title",
        "section": "Introduction"
    }
]
pipeline.ingest_documents(documents)

# Query with retrieval, reranking, and answer generation
response = pipeline.query(
    question="What is machine learning?",
    top_k=10,
    rerank_top_k=5,
    use_mmr=True
)

# Display answer with citations
print(response["answer"])
for source in response["sources"]:
    print(f"[{source['citation']}] {source['title']}")
```

### Individual Components
```python
from vector_db_setup import VectorDatabaseManager

# Initialize
db = VectorDatabaseManager()

# Upsert vectors
vectors = [
    ("id1", embedding1, {"text": "sample 1"}),
    ("id2", embedding2, {"text": "sample 2"}),
]
db.upsert_vectors(vectors)

# Query similar vectors
results = db.query_vectors(
    query_vector=query_embedding,
    top_k=5
)
```

## Documentation

### Quick Start
- [README.md](README.md) - This file
- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Complete step-by-step installation guide

### Complete Guides
- [RAG_CONFIGURATION.md](RAG_CONFIGURATION.md) - Complete RAG pipeline setup
- [VECTOR_DB_DOCUMENTATION.md](VECTOR_DB_DOCUMENTATION.md) - Vector database details

### Components
- **Embeddings**: FastEmbed (local, free)
- **Chunking**: 800-1200 tokens with 10-15% overlap
- **Vector DB**: Pinecone (hosted, serverless)
- **Retrieval**: MMR for diversity
- **Reranking**: FlashRank (local, free)
- **LLM**: Groq Llama-3.1-8b (fast, free tier)

## Requirements

- Python 3.8+
- Pinecone account (free tier available) - https://app.pinecone.io
- Groq API key (free tier available) - https://console.groq.com

## Support

For issues or questions:
- Pinecone Docs: https://docs.pinecone.io/
- OpenAI Docs: https://platform.openai.com/docs/

# 🚀 RAG Q&A System - Track B Submission

### 🔗 Project Links
* **Live App URL:** [https://rag-qa-system-main.streamlit.app/](https://rag-app-system-main.streamlit.app/)
* **Resume Link:** [Click here to view my Resume](https://drive.google.com/file/d/1WuMTdeW3MFe5GI82d_x7aM-5gc_nsim3/view?usp=drivesdk)
* **GitHub Repository:** [https://github.com/sonam869/rag-qa-system-main](https://github.com/sonam869/rag-qa-system-main)

---

## 📝 Remarks
- **Trade-offs:** Used local embeddings (FastEmbed) and reranking (FlashRank) to minimize API latency and costs.
- **Limits:** The current Pinecone serverless index is limited to 2GB of storage.
- **Future Improvements:** Implementation of a "Self-RAG" loop to verify hallucination before showing the answer.
