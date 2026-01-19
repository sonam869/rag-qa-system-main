# Vector Database Documentation

## Overview
This project uses **Pinecone** as a hosted vector database for storing and querying embeddings.

## Configuration Details

### Index Information

| Property | Value | Description |
|----------|-------|-------------|
| **Index Name** | `tumpa-embeddings` | Unique identifier for the vector index |
| **Dimensionality** | `1536` | Vector dimension (compatible with OpenAI text-embedding-ada-002) |
| **Distance Metric** | `cosine` | Similarity measure for vector comparisons |
| **Cloud Provider** | `AWS` | Hosting infrastructure |
| **Region** | `us-east-1` | Data center location |
| **Deployment Type** | `Serverless` | Auto-scaling, pay-per-use model |

### Dimensionality

**1536 dimensions** - This is the standard output dimension for OpenAI's `text-embedding-ada-002` model, which produces high-quality embeddings for text data. If you plan to use a different embedding model, adjust the dimension accordingly:

- OpenAI ada-002: 1536
- OpenAI text-embedding-3-small: 1536
- OpenAI text-embedding-3-large: 3072
- Cohere embed-english-v3.0: 1024
- Sentence Transformers (all-MiniLM-L6-v2): 384

## Upsert Strategy

### Overview
The upsert operation combines "update" and "insert" - it will update existing vectors or insert new ones based on the vector ID.

### Implementation Details

1. **Batch Processing**
   - Vectors are upserted in batches of 100
   - This optimizes network usage and API rate limits
   - Reduces overall latency for large datasets

2. **Vector Format**
   ```python
   vectors = [
       (
           "unique_id",           # Unique identifier (string)
           [0.1, 0.2, ...],       # Embedding vector (list of floats, length=1536)
           {"text": "content"}    # Metadata (optional dict)
       )
   ]
   ```

3. **Metadata Storage**
   - Store original text, source file, timestamps, etc.
   - Metadata is returned with query results
   - Useful for filtering and retrieving original content

4. **ID Strategy**
   - Use consistent, meaningful IDs (e.g., `doc_123`, `chunk_456`)
   - IDs should be unique across your dataset
   - Recommended format: `{source}_{identifier}_{chunk_number}`

### Performance Considerations

- **Batch Size**: 100 vectors per batch (optimal for most use cases)
- **Parallelization**: Can be increased for larger datasets
- **Rate Limits**: Serverless tier handles up to 100 requests/second
- **Retry Logic**: Built-in retry for transient failures

## Usage Workflow

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your PINECONE_API_KEY
```

### 2. Initialize Database
```bash
python vector_db_setup.py
```

### 3. Upsert Vectors
```python
from vector_db_setup import VectorDatabaseManager

db = VectorDatabaseManager()
vectors = [
    ("id1", embedding1, {"text": "sample text 1"}),
    ("id2", embedding2, {"text": "sample text 2"}),
]
db.upsert_vectors(vectors)
```

### 4. Query Similar Vectors
```python
results = db.query_vectors(
    query_vector=query_embedding,
    top_k=5,
    include_metadata=True
)
```

## API Keys and Security

### Required Environment Variables
- `PINECONE_API_KEY`: Your Pinecone API key (get from https://app.pinecone.io)
- `OPENAI_API_KEY`: (Optional) For generating embeddings

### Security Best Practices
1. Never commit `.env` file to version control
2. Use `.gitignore` to exclude sensitive files
3. Rotate API keys periodically
4. Use read-only keys for production queries when possible

## Cost Estimation

### Pinecone Serverless Pricing (as of 2026)
- **Storage**: ~$0.40 per GB per month
- **Read Units**: $0.10 per 100k reads
- **Write Units**: $2.00 per 1M writes

### Example Cost Calculation
For 1 million vectors (1536 dimensions):
- Storage: ~6 GB = $2.40/month
- 1M writes (initial upload): $2.00 one-time
- 100k queries/month: $0.10/month

**Total First Month**: ~$4.50
**Ongoing**: ~$2.50/month + query costs

## Monitoring and Maintenance

### Index Statistics
```python
stats = db.get_index_stats()
print(f"Total vectors: {stats.total_vector_count}")
```

### Health Checks
- Monitor query latency (should be < 100ms for p95)
- Check upsert success rates
- Track storage usage

## Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure `PINECONE_API_KEY` is set in `.env`
   - Verify key is active in Pinecone dashboard

2. **Dimension Mismatch**
   - All vectors must have exactly 1536 dimensions
   - Check your embedding model output

3. **Index Not Found**
   - Run `vector_db_setup.py` to create the index
   - Verify index name matches configuration

4. **Rate Limiting**
   - Reduce batch size or add delays
   - Consider upgrading to higher tier

## Additional Resources

- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Vector Database Best Practices](https://www.pinecone.io/learn/vector-database/)
