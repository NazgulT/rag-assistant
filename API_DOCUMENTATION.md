# RAG System API Documentation

## Authentication
Currently, the API does not require authentication. Future versions will include JWT-based authentication.

## Base URL
```
http://localhost:8000
```

## Response Format
All responses are in JSON format.

### Success Response (2xx)
```json
{
  "status": "success",
  "data": {},
  "message": "Operation completed"
}
```

### Error Response (4xx, 5xx)
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Endpoints

### 1. Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "rag_system": "initialized"
}
```

---

### 2. System Information
```
GET /api/v1/info
```

**Response:**
```json
{
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dimension": 384,
  "llm_model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  "retrieval_type": "hybrid",
  "reranker_model": "cross-encoder/mmarco-MiniLMv2-L12-H384-v1"
}
```

---

### 3. Collection Statistics
```
GET /api/v1/collections/stats
```

**Response:**
```json
{
  "collection_name": "rag_documents",
  "total_chunks": 150,
  "persist_directory": "/path/to/chroma_db"
}
```

---

### 4. Ingest File
```
POST /api/v1/documents/ingest-file
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (required): The file to upload (PDF, CSV, DOCX, TXT)
- `metadata` (optional): JSON string of additional metadata

**Request Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/ingest-file" \
  -F "file=@document.pdf" \
  -F "metadata={\"author\": \"John Doe\"}"
```

**Response:**
```json
{
  "document_id": "doc_a1b2c3d4e5f6g7h8",
  "chunks_created": 25,
  "embeddings_created": 25,
  "processing_time": 4.23
}
```

---

### 5. Ingest URL
```
POST /api/v1/documents/ingest-url
Content-Type: application/json
```

**Request Body:**
```json
{
  "url": "https://example.com/article",
  "metadata": {
    "source_type": "blog",
    "date": "2026-03-07"
  }
}
```

**Response:**
```json
{
  "document_id": "doc_x9y8z7w6v5u4t3s2",
  "chunks_created": 18,
  "embeddings_created": 18,
  "processing_time": 2.15
}
```

---

### 6. Ingest Text
```
POST /api/v1/documents/ingest-text
Content-Type: application/json
```

**Request Body:**
```json
{
  "file_path": "This is the document content...",
  "metadata": {
    "source": "user_input"
  }
}
```

**Response:**
```json
{
  "document_id": "doc_m7n6o5p4q3r2s1t0",
  "chunks_created": 5,
  "embeddings_created": 5,
  "processing_time": 0.89
}
```

---

### 7. Query (Complete RAG Pipeline)
```
POST /api/v1/query
Content-Type: application/json
```

**Query Parameters:**
- `k_retrieve` (optional, default: 10): Number of documents to retrieve
- `k_rerank` (optional, default: 5): Number of documents to rerank
- `use_reranking` (optional, default: true): Enable reranking

**Request Body:**
```json
{
  "text": "What is machine learning?",
  "query_id": "query_001",
  "filters": null
}
```

**Request Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/query?k_retrieve=15&k_rerank=5&use_reranking=true" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is machine learning?"
  }'
```

**Response:**
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence...",
  "retrieved_documents": [
    {
      "chunk_id": "chunk_001",
      "content": "Machine learning is...",
      "document_id": "doc_001",
      "score": 0.92,
      "metadata": {}
    }
  ],
  "reranked_documents": [
    {
      "chunk_id": "chunk_001",
      "content": "Machine learning is...",
      "document_id": "doc_001",
      "original_score": 0.92,
      "reranked_score": 0.95,
      "rank": 1
    }
  ],
  "generation_time": 3.45,
  "retrieval_time": 0.56,
  "total_time": 4.01
}
```

---

### 8. Retrieve Documents
```
POST /api/v1/retrieve
Content-Type: application/json
```

**Query Parameters:**
- `k` (optional, default: 10): Number of documents to retrieve

**Request Body:**
```json
{
  "text": "What is Python?"
}
```

**Response:**
```json
{
  "count": 5,
  "results": [
    {
      "chunk_id": "chunk_001",
      "content": "Python is a programming language...",
      "document_id": "doc_001",
      "score": 0.87,
      "metadata": {}
    }
  ]
}
```

---

### 9. Rerank Documents
```
POST /api/v1/rerank
Content-Type: application/json
```

**Query Parameters:**
- `k` (optional, default: 5): Number of documents to return

**Request Body:**
```json
{
  "text": "What is machine learning?",
  "documents": [
    {
      "chunk_id": "chunk_001",
      "content": "ML is...",
      "score": 0.8
    }
  ]
}
```

**Response:**
```json
{
  "count": 1,
  "results": [
    {
      "chunk_id": "chunk_001",
      "content": "ML is...",
      "original_score": 0.8,
      "reranked_score": 0.92,
      "rank": 1
    }
  ]
}
```

---

### 10. Evaluate Answer
```
POST /api/v1/evaluate
Content-Type: application/json
```

**Query Parameters:**
- `query` (required): The original query
- `answer` (required): The generated answer
- `contexts` (required): List of context passages
- `ground_truth` (optional): The ground truth answer

**Request Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI?",
    "answer": "AI is artificial intelligence...",
    "contexts": ["AI is a field of computer science..."],
    "ground_truth": "AI is the simulation of human intelligence..."
  }'
```

**Response:**
```json
{
  "answer_relevancy": 0.92,
  "faithfulness": 0.88,
  "context_precision": 0.85,
  "context_recall": 0.90,
  "average_score": 0.89
}
```

---

## Error Handling

### Common Errors

**503 Service Unavailable**
```json
{
  "detail": "RAG system not initialized"
}
```

**400 Bad Request**
```json
{
  "detail": "URL is required"
}
```

**500 Internal Server Error**
```json
{
  "detail": "Error message describing the issue"
}
```

---

## Rate Limiting

Currently not implemented. Future versions will include rate limiting.

---

## Pagination

Not applicable to current endpoints. Future versions may implement pagination for large result sets.

---

## Caching

Responses are cached for 1 hour by default. Clear cache by restarting the service.

---

## Batch Operations

For batch queries, use the evaluation endpoint multiple times or implement batching in your client.

---

## Monitoring

- Use the `/health` endpoint for health checks
- View API logs in `logs/rag_system.log`
- Monitor MLFlow at `http://localhost:5000`

---

## Examples

### Python Client
```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Query
response = requests.post(
    f"{BASE_URL}/query",
    json={"text": "What is machine learning?"}
)
result = response.json()
print(result["answer"])

# Evaluate
eval_response = requests.post(
    f"{BASE_URL}/evaluate",
    json={
        "query": "What is ML?",
        "answer": result["answer"],
        "contexts": [doc["content"] for doc in result["reranked_documents"]]
    }
)
metrics = eval_response.json()
print(f"Answer Relevancy: {metrics['answer_relevancy']:.4f}")
```

### JavaScript Client
```javascript
const BASE_URL = 'http://localhost:8000/api/v1';

async function queryRAG(question) {
  const response = await fetch(`${BASE_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: question })
  });
  return await response.json();
}

const result = await queryRAG('What is machine learning?');
console.log(result.answer);
```

---

## WebSocket Support

Not currently implemented. Consider for real-time streaming responses in future versions.
