# Complete Installation & Usage Guide

This guide walks you through setting up and using the RAG Q&A system from scratch.

## 📋 Prerequisites

- Python 3.8 or higher
- Windows, macOS, or Linux
- Internet connection for API access

## 🚀 Step-by-Step Installation

### Step 1: Clone or Download the Project

```bash
# If you have git
git clone <repository-url>
cd tumpa

# Or download and extract the ZIP file, then navigate to the folder
cd tumpa
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal.

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

This will install:
- Pinecone (vector database)
- FastEmbed (local embeddings)
- FlashRank (reranking)
- Groq (LLM)
- Streamlit (web interface)
- And other dependencies

### Step 4: Get API Keys

#### 4.1 Pinecone API Key (Required)

1. Go to [https://app.pinecone.io](https://app.pinecone.io)
2. Sign up for a free account
3. Create a new API key
4. Copy the API key (starts with `pcsk_...`)

#### 4.2 Groq API Key (Required)

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the API key (starts with `gsk_...`)

### Step 5: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# On Windows:
copy .env.example .env
```

Open `.env` file in a text editor and add your API keys:

```env
# Pinecone API Key
PINECONE_API_KEY="pcsk_your_actual_key_here"

# Groq API Key
GROQ_API_KEY="gsk_your_actual_key_here"

# OpenAI API Key (Optional - not needed for this setup)
OPENAI_API_KEY=your_openai_api_key_here
```

**Important:** Replace the placeholder values with your actual API keys!

### Step 6: Initialize Vector Database

```bash
python vector_db_setup.py
```

This will:
- Connect to Pinecone
- Create an index named `tumpa-embeddings`
- Configure it with 384 dimensions (for FastEmbed)

Expected output:
```
Creating index 'tumpa-embeddings'...
Index 'tumpa-embeddings' created successfully!

Index Statistics:
Total vectors: 0
Dimension: 384
Index name: tumpa-embeddings

✅ Vector database setup completed successfully!
```

## 🎯 Usage

### Option 1: Web Interface (Recommended)

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

#### Using the Web Interface:

**1. Upload Documents (Tab 1)**
   - Choose "Upload File" or "Paste Text"
   - If uploading files:
     - Click "Browse files"
     - Select .txt, .md, or .pdf files
     - Multiple files supported
   - If pasting text:
     - Enter document title
     - Paste or type your text
   - Click "📚 Process & Ingest Documents"

**2. Ask Questions (Tab 2)**
   - Type your question in the query box
   - Click "🔍 Get Answer"
   - View the answer with inline citations [1], [2], etc.
   - Expand source citations to see:
     - Original text snippet
     - Source file name
     - Relevance score
     - Section information

**3. Adjust Settings (Sidebar)**
   - Number of chunks to retrieve (5-30)
   - Top chunks after reranking (3-10)
   - Enable/disable MMR (diversity)
   - Adjust MMR lambda (relevance vs diversity)
   - LLM temperature (creativity)
   - Max tokens (response length)

**4. View History (Tab 3)**
   - Review all previous queries
   - See past answers and citations
   - Clear history when needed

### Option 2: Python Scripts

#### Test Individual Components:

```bash
# Test embeddings
python embeddings.py

# Test chunking
python chunking.py

# Test retriever
python retriever.py

# Test reranker
python reranker.py

# Test LLM answering
python llm_answering.py
```

#### Use the Complete Pipeline:

```bash
python rag_pipeline.py
```

This runs a demo with sample documents and queries.

#### Use in Your Own Code:

```python
from rag_pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline()

# Ingest documents
documents = [
    {
        "text": "Your document content here...",
        "source": "document.pdf",
        "title": "Document Title",
        "section": "Chapter 1"
    }
]
pipeline.ingest_documents(documents)

# Ask questions
response = pipeline.query(
    question="What is the main topic?",
    top_k=10,
    rerank_top_k=5,
    use_mmr=True
)

# Get answer and sources
print("Answer:", response["answer"])
for source in response["sources"]:
    print(f"[{source['citation']}] {source['title']}")
```

## 📊 System Architecture

```
User Question
    ↓
1. Embedding Generation (FastEmbed - Local)
    ↓
2. Vector Retrieval (Pinecone - Top 10 similar chunks)
    ↓
3. MMR Filtering (Diversity)
    ↓
4. Reranking (FlashRank - Top 5 most relevant)
    ↓
5. Answer Generation (Groq Llama-3.1-8b)
    ↓
Answer + Citations + Sources
```

## 🔧 Configuration Options

### Chunking Settings (chunking.py)
- **Chunk size**: 800-1200 tokens (default: 1000)
- **Overlap**: 10-15% (default: 12.5%)
- Adjust in `TextChunker(chunk_size=1000, overlap_percentage=12.5)`

### Embedding Model (embeddings.py)
- **Default**: BAAI/bge-small-en-v1.5 (384 dim)
- **Alternatives**: 
  - BAAI/bge-base-en-v1.5 (768 dim) - better quality
  - sentence-transformers/all-MiniLM-L6-v2 (384 dim)

### Retrieval Settings
- **top_k**: Number of chunks to retrieve (default: 10)
- **fetch_k**: Candidates for MMR (default: 50)
- **lambda**: MMR balance 0-1 (default: 0.5)
  - 1.0 = pure relevance
  - 0.0 = pure diversity

### Reranking Model (reranker.py)
- **Default**: ms-marco-MiniLM-L-12-v2
- **Alternatives**:
  - ms-marco-MultiBERT-L-12 (multilingual)
  - rank-T5-flan (more accurate)

### LLM Settings
- **Model**: llama-3.1-8b-instant (default)
- **Temperature**: 0.3 (default) - lower = more focused
- **Max tokens**: 1024 (default)

## 🐛 Troubleshooting

### Issue: "PINECONE_API_KEY not found"
**Solution:** Make sure you created `.env` file and added your API key

### Issue: "Module not found"
**Solution:** Activate virtual environment and run `pip install -r requirements.txt`

### Issue: "Index already exists"
**Solution:** That's fine! The system will use the existing index

### Issue: Streamlit won't start
**Solution:** 
```bash
# Deactivate and reactivate venv
deactivate
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # macOS/Linux

# Reinstall streamlit
pip install streamlit

# Try again
streamlit run app.py
```

### Issue: Slow embedding generation
**Solution:** First run downloads models (~100MB). Subsequent runs are fast.

### Issue: PDF text extraction incomplete
**Solution:** Some PDFs with complex layouts may not extract perfectly. Try converting to text first.

## 📝 Example Workflow

### Complete Example Session:

```bash
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Launch web interface
streamlit run app.py

# 3. In the browser:
#    - Go to "Document Upload" tab
#    - Upload your PDF/text files
#    - Click "Process & Ingest Documents"
#    - Wait for confirmation

# 4. Go to "Ask Questions" tab
#    - Type: "What are the main topics discussed?"
#    - Click "Get Answer"
#    - Review answer with citations

# 5. Adjust settings in sidebar if needed
#    - Increase top_k for more context
#    - Decrease temperature for more focused answers

# 6. Check "History" tab to review past queries
```

## 💡 Tips for Best Results

### Document Preparation
- ✅ Clean, well-formatted text works best
- ✅ Include document titles and sections
- ✅ Break very long documents into chapters
- ❌ Avoid scanned images (OCR needed)

### Querying
- ✅ Ask specific questions
- ✅ Use keywords from your documents
- ✅ Try different phrasings if unsatisfied
- ❌ Avoid overly broad questions

### Settings Tuning
- **More context needed?** → Increase top_k
- **Too much redundancy?** → Enable MMR, lower lambda
- **Answers too creative?** → Lower temperature
- **Better relevance?** → Increase rerank_top_k

## 📚 Additional Resources

- [RAG_CONFIGURATION.md](RAG_CONFIGURATION.md) - Detailed configuration guide
- [VECTOR_DB_DOCUMENTATION.md](VECTOR_DB_DOCUMENTATION.md) - Vector database details
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Groq Documentation](https://console.groq.com/docs)
- [FastEmbed Documentation](https://qdrant.github.io/fastembed/)

## 🆘 Getting Help

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review error messages carefully
3. Ensure all API keys are correctly set
4. Verify virtual environment is activated
5. Check that all dependencies are installed

## 🎉 You're Ready!

Your RAG Q&A system is now set up and ready to use. Start by:
1. Uploading some documents
2. Asking questions
3. Exploring the settings to optimize results

Happy querying! 🚀
