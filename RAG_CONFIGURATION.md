# RAG System Configuration Guide

## Complete RAG Pipeline Components

### 1. **Embeddings & Chunking** ✅

#### Embedding Model: FastEmbed (Local)
- **Model**: `BAAI/bge-small-en-v1.5`
- **Dimension**: 384
- **Type**: Local (no API calls)
- **Benefits**: Fast, free, privacy-friendly

#### Chunking Strategy
- **Size**: 800-1,200 tokens (default: 1,000)
- **Overlap**: 10-15% (default: 12.5% = 125 tokens)
- **Encoding**: cl100k_base (GPT-3.5/GPT-4 tokenizer)

#### Metadata Storage
Each chunk stores:
- `source`: Source file/document identifier
- `title`: Document title
- `section`: Section name (optional)
- `position`: Chunk position in document
- `start_char`: Starting character position
- `end_char`: Ending character position
- `token_count`: Number of tokens in chunk
- Custom metadata fields (author, date, etc.)

### 2. **Vector Database** ✅

#### Configuration
- **Provider**: Pinecone (Hosted)
- **Index Name**: `tumpa-embeddings`
- **Dimension**: 384 (matches FastEmbed)
- **Metric**: Cosine similarity
- **Deployment**: Serverless (AWS us-east-1)

### 3. **Retriever + Reranker** ✅

#### Retrieval with MMR
- **Algorithm**: Maximal Marginal Relevance
- **Purpose**: Balance relevance and diversity
- **Parameters**:
  - `top_k`: Final number of results (default: 10)
  - `fetch_k`: Candidates to fetch (default: 50)
  - `lambda`: Balance parameter (default: 0.5)

**MMR Formula**:
```
MMR = λ × Relevance - (1-λ) × MaxSimilarity
```

#### Reranker: FlashRank (Free)
- **Model**: `ms-marco-MiniLM-L-12-v2`
- **Type**: Local, free
- **Purpose**: Improve ranking quality
- **Input**: Retrieved candidates
- **Output**: Reranked results with scores

### 4. **LLM & Answering** ✅

#### LLM Provider: Groq
- **Model**: `llama-3.1-8b-instant`
- **Benefits**: Fast inference, free tier
- **Temperature**: 0.3 (focused, consistent)
- **Max tokens**: 1,024

#### Answer Generation
- **Grounding**: Only uses provided context
- **Citations**: Inline references [1], [2], etc.
- **No-answer handling**: Gracefully indicates when context is insufficient
- **Source mapping**: Each citation linked to source chunk

#### Citation Format
```
Answer text with citation [1] and another fact [2].

Sources:
[1] Document Title - Section
    Source: file.txt
    Text: Original text snippet...
    
[2] Another Document - Section
    Source: doc2.txt
    Text: Another snippet...
```

## Pipeline Flow

```
1. Document Ingestion
   ├─ Chunk text (1000 tokens, 12.5% overlap)
   ├─ Generate embeddings (FastEmbed)
   └─ Store in Pinecone with metadata

2. Query Processing
   ├─ Generate query embedding
   ├─ Retrieve top-k with MMR (diversity)
   ├─ Rerank with FlashRank
   ├─ Generate answer with Groq
   └─ Return answer + citations + sources
```

## File Structure

```
tumpa/
├── embeddings.py              # FastEmbed local embeddings
├── chunking.py                # Text chunking with metadata
├── vector_db_setup.py         # Pinecone database setup
├── retriever.py               # MMR retrieval
├── reranker.py                # FlashRank reranking
├── llm_answering.py           # Groq answer generation
├── rag_pipeline.py            # Complete pipeline
├── requirements.txt           # All dependencies
├── .env.example              # Environment template
└── RAG_CONFIGURATION.md      # This file
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add:
# - PINECONE_API_KEY (from https://app.pinecone.io)
# - GROQ_API_KEY (from https://console.groq.com)
```

### 3. Initialize Vector Database
```bash
python vector_db_setup.py
```

### 4. Run Complete Pipeline
```bash
python rag_pipeline.py
```

## Usage Example

```python
from rag_pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline()

# Ingest documents
documents = [
    {
        "text": "Your document content...",
        "source": "document.pdf",
        "title": "Document Title",
        "section": "Introduction"
    }
]
pipeline.ingest_documents(documents)

# Query
response = pipeline.query(
    question="What is machine learning?",
    top_k=10,           # Retrieve 10 chunks
    rerank_top_k=5,     # Rerank to top 5
    use_mmr=True        # Use MMR for diversity
)

# Display
print(response["answer"])
for source in response["sources"]:
    print(f"[{source['citation']}] {source['title']}")
```

## Performance Characteristics

### Embedding Generation (FastEmbed)
- **Speed**: ~100-500 texts/second
- **Cost**: Free (local)
- **Quality**: High (SOTA model)

### Vector Retrieval (Pinecone)
- **Latency**: <100ms (p95)
- **Scale**: Millions of vectors
- **Cost**: ~$2-5/month for small projects

### Reranking (FlashRank)
- **Speed**: ~50-100 pairs/second
- **Cost**: Free (local)
- **Improvement**: 10-20% better relevance

### Answer Generation (Groq)
- **Speed**: ~300 tokens/second
- **Cost**: Free tier available
- **Quality**: High (Llama-3.1-8b)

## Customization Options

### Adjust Chunk Size
```python
chunker = TextChunker(
    chunk_size=1200,        # Larger chunks
    overlap_percentage=10   # Less overlap
)
```

### Change Embedding Model
```python
embedder = EmbeddingGenerator(
    model_name="BAAI/bge-base-en-v1.5"  # 768 dimensions
)
```

### Tune MMR Balance
```python
# More relevance, less diversity
results = retriever.retrieve_with_mmr(
    query=query,
    lambda_param=0.7
)

# More diversity, less relevance
results = retriever.retrieve_with_mmr(
    query=query,
    lambda_param=0.3
)
```

### Use Different LLM
```python
generator = GroqAnswerGenerator(
    model="llama-3.1-70b-versatile"  # Larger model
)
```

## Best Practices

### Document Ingestion
1. Clean and preprocess text before ingestion
2. Add rich metadata for better filtering
3. Use meaningful source identifiers
4. Batch large ingestions for efficiency

### Retrieval
1. Start with `top_k=20-30` for MMR
2. Use `lambda=0.5` for balanced results
3. Enable MMR for diverse results
4. Use metadata filters when applicable

### Reranking
1. Rerank to top 3-5 results for LLM context
2. Set score threshold to filter low-quality matches
3. Use faster models for real-time applications

### Answer Generation
1. Keep temperature low (0.2-0.3) for factual answers
2. Provide clear system prompts
3. Always cite sources
4. Handle edge cases (no results, errors)

## Monitoring & Debugging

### Check Pipeline Stats
```python
stats = pipeline.get_stats()
print(stats)
```

### Test Individual Components
```bash
python embeddings.py        # Test embeddings
python chunking.py          # Test chunking
python retriever.py         # Test retrieval
python reranker.py          # Test reranking
python llm_answering.py     # Test LLM
```

### Common Issues

**Issue**: Dimension mismatch
- **Solution**: Ensure embedding dimension matches Pinecone index

**Issue**: Low relevance scores
- **Solution**: Adjust chunk size, use better overlap, improve metadata

**Issue**: Slow retrieval
- **Solution**: Reduce top_k, optimize index, use filters

**Issue**: Poor answer quality
- **Solution**: Increase rerank_top_k, tune prompts, use better LLM

## Cost Estimation

### Monthly Costs (1M queries)
- **Pinecone**: ~$5-10 (serverless)
- **FastEmbed**: $0 (local)
- **FlashRank**: $0 (local)
- **Groq**: $0-20 (free tier + overage)
- **Total**: ~$5-30/month

### Optimization Tips
1. Use local models (FastEmbed, FlashRank) to reduce API costs
2. Batch operations when possible
3. Cache frequent queries
4. Use Pinecone serverless for auto-scaling

## Additional Resources

- [FastEmbed Documentation](https://qdrant.github.io/fastembed/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [FlashRank GitHub](https://github.com/PrithivirajDamodaran/FlashRank)
- [Groq Documentation](https://console.groq.com/docs)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
