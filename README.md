# Production RAG Assistant

Production-ready modular Retrieval-Augmented Generation (RAG) platform built using Python and modern ML infrastructure. The system is designed using OOP architecture, hybrid retrieval strategies, and evaluation observability to ensure reliable performance. 

## System Architecture
...goes here...

### Component Details

#### 1. **Document Ingestion Module** (`app/ingestion/`)
- **PDFLoader**: Extracts text from PDF files
- **CSVLoader**: Converts CSV data to structured text
- **DocxLoader**: Extracts text from Word documents
- **TextLoader**: Loads plain text files
- **URLLoader**: Fetches and parses web content
- **DataFrameLoader**: Ingests pandas DataFrames

#### 2. **Embeddings Module** (`app/embeddings/`)
- Uses **Sentence Transformers** for semantic embeddings
- Generates 384-dimensional embeddings (all-MiniLM-L6-v2)
- Batch processing for efficiency
- Device-agnostic (CPU, GPU, MPS)

#### 3. **Vector Storage** (`app/storage/`)
- **ChromaDB** with persistent storage
- HNSW indexing for fast similarity search
- Cosine distance metric
- Metadata support for filtering
- Collection management

#### 4. **Retrieval Module** (`app/retrieval/`)
- **Semantic Retriever**: Vector similarity search
- **BM25 Retriever**: Keyword-based search
- **Hybrid Retriever**: Combines semantic + BM25 with configurable weights
  - Default: 60% semantic, 40% BM25
  - Score normalization and fusion

#### 5. **Reranking Module** (`app/reranking/`)
- **CrossEncoderReranker**: Uses cross-encoder models for fine-grained relevance scoring
- Re-ranks top-k documents for improved quality
- Threshold-based filtering

#### 6. **LLM Generation Module** (`app/generation/`)
- **HuggingFace Transformers** integration
- **SmolLM2-1.7B-Instruct** for lightweight inference
- Context-aware prompt engineering
- Configurable temperature and max tokens
- Supports CPU, GPU, and MPS devices

#### 7. **Evaluation Module** (`app/evaluation/`)
- **RAGAS Metrics Integration**
  - Answer Relevancy
  - Faithfulness
  - Context Precision
  - Context Recall
- Batch evaluation support
- Aggregate metric computation

#### 8. **Experiment Tracking** (`app/logging/`)
- **MLFlow Integration**
  - Run tracking
  - Parameter logging
  - Metric logging
  - Artifact storage

#### 9. **Configuration Management** (`app/config/`)
- Centralized settings using Pydantic
- Environment variable support
- Sensible defaults for all parameters

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
```bash
cd /Projects/src/rag-assistant-fastapi
```

2. **Create virtual environment**
```bash
python -m venv .rag
source .rag/bin/activate  # On macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

#### 1. **As a Library**

```python
from app.rag_system import RAGSystem

# Initialize RAG system
rag = RAGSystem(use_mlflow=True)

# Ingest a document
result = rag.ingest_document(
    source="path/to/document.pdf",
    source_type="file",
    metadata={"author": "John Doe"}
)

# Query the system
response = rag.answer_query("What is machine learning?")

print(f"Answer: {response.answer}")
print(f"Retrieved {len(response.retrieved_documents)} documents")
print(f"Reranked {len(response.reranked_documents)} documents")
```

#### 2. **Run Example Script**

```bash
python example.py
```

#### 3. **Start FastAPI Server**

```bash
python main.py
```

The API will be available at `http://localhost:8000`

- API Documentation: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## 📚 API Endpoints

### Health & Info
- `GET /health` - Health check
- `GET /api/v1/info` - System information
- `GET /api/v1/collections/stats` - Collection statistics

### Document Ingestion
- `POST /api/v1/documents/ingest-file` - Ingest uploaded file
- `POST /api/v1/documents/ingest-url` - Ingest web URL
- `POST /api/v1/documents/ingest-text` - Ingest plain text

### Retrieval & Generation
- `POST /api/v1/query` - Complete RAG pipeline (retrieve → rerank → generate)
- `POST /api/v1/retrieve` - Retrieve documents only
- `POST /api/v1/rerank` - Rerank documents

### Evaluation
- `POST /api/v1/evaluate` - Evaluate answer with RAGAS metrics

## ⚙️ Configuration

Edit `app/config/settings.py` to customize:

```python
# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Retrieval
N_RETRIEVE = 10
RETRIEVAL_TYPE = "hybrid"  # hybrid, semantic, bm25
BM25_K1 = 1.5
BM25_B = 0.75

# Reranking
RERANKER_MODEL = "cross-encoder/mmarco-MiniLMv2-L12-H384-v1"
N_RERANK = 5

# LLM
LLM_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
LLM_DEVICE = "mps"  # cpu, cuda, mps
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 512

# Document Processing
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
```

## 📊 Performance Characteristics

| Component | Model Size | Performance | Device |
|-----------|-----------|-------------|--------|
| Embeddings | 22M | ~1000 texts/s | CPU/GPU/MPS |
| Retrieval (Semantic) | N/A | <100ms | RAM |
| Retrieval (BM25) | N/A | <50ms | RAM |
| Reranking | 33M | ~500ms for 10 docs | CPU/GPU |
| LLM Generation | 1.7B | ~2-5s per query | CPU/MPS |

## 📁 Project Structure

```
rag-assistant-fastapi/
├── app/
│   ├── __init__.py
│   ├── config/
│   │   └── settings.py              # Central configuration
│   ├── models/
│   │   └── schemas.py               # Pydantic models
│   ├── ingestion/
│   │   └── loaders.py               # Document loaders
│   ├── embeddings/
│   │   └── embedding.py             # Embedding generation
│   ├── storage/
│   │   └── chroma_store.py          # Vector storage
│   ├── retrieval/
│   │   └── retriever.py             # Retrieval strategies
│   ├── reranking/
│   │   └── reranker.py              # Reranking models
│   ├── generation/
│   │   └── generator.py             # LLM generation
│   ├── evaluation/
│   │   └── evaluator.py             # RAGAS evaluation
│   ├── logging/
│   │   ├── logger.py                # Logging setup
│   │   └── mlflow_tracker.py        # MLFlow tracking
│   ├── utils/
│   │   └── helpers.py               # Utility functions
│   ├── api/
│   │   └── main.py                  # FastAPI application
│   └── rag_system.py                # Main orchestrator
├── data/
│   └── chroma_db/                   # Persistent vector store
├── logs/
│   └── rag_system.log               # Application logs
├── mlruns/                          # MLFlow experiments
├── main.py                          # Entry point
├── example.py                       # Example usage
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🔍 Detailed Usage Examples

### Example 1: Ingest Multiple Documents

```python
from app.rag_system import RAGSystem
import pandas as pd

rag = RAGSystem()

# Ingest PDF
rag.ingest_document(
    source="document.pdf",
    source_type="file"
)

# Ingest from URL
rag.ingest_document(
    source="https://example.com/article",
    source_type="url"
)

# Ingest from DataFrame
df = pd.read_csv("data.csv")
rag.ingest_document(
    source=df,
    source_type="dataframe"
)

# Ingest plain text
rag.ingest_document(
    source="This is some content...",
    source_type="text"
)
```

### Example 2: Custom Retrieval Configuration

```python
response = rag.answer_query(
    query="What are the benefits?",
    k_retrieve=20,          # Retrieve more documents
    k_rerank=5,             # Rerank to top 5
    use_reranking=True      # Enable reranking
)
```

### Example 3: Evaluation Pipeline

```python
# Get RAG response
response = rag.answer_query("What is AI?")

# Evaluate the answer
eval_result = rag.evaluate_answer(
    query="What is AI?",
    answer=response.answer,
    contexts=[doc.content for doc in response.reranked_documents],
    ground_truth="AI is the simulation of human intelligence..."
)

print(f"Answer Relevancy: {eval_result['answer_relevancy']:.4f}")
print(f"Faithfulness: {eval_result['faithfulness']:.4f}")
print(f"Context Precision: {eval_result['context_precision']:.4f}")
print(f"Context Recall: {eval_result['context_recall']:.4f}")
```

### Example 4: MLFlow Tracking

```python
rag = RAGSystem(use_mlflow=True)

# MLFlow automatically tracks:
# - Ingestion parameters and metrics
# - Query processing times
# - Retrieval/reranking statistics
# - Evaluation metrics

# View experiments
mlflow ui --backend-store-uri mlruns
```

## 🛠️ Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run tests
pytest tests/ -v
```

### Adding Custom Models

1. **Custom Embeddings**
```python
from app.embeddings.embedding import EmbeddingModel

class CustomEmbedding(EmbeddingModel):
    def __init__(self, model_name):
        self.model = load_custom_model(model_name)
    
    def encode(self, texts):
        return self.model.encode(texts)
    
    def get_dimension(self):
        return 768
```

2. **Custom Reranker**
```python
from app.reranking.reranker import Reranker

class CustomReranker(Reranker):
    def rerank(self, query, documents, k=5):
        # Your reranking logic
        pass
```

## 📈 Monitoring & Logging

### Log Levels

```python
# In app/config/settings.py
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### View Logs

```bash
# Real-time logs
tail -f logs/rag_system.log

# Grep for errors
grep "ERROR" logs/rag_system.log
```

### MLFlow Dashboard

```bash
cd /path/to/rag-assistant-fastapi
mlflow ui
# Visit http://localhost:5000
```

## 🔐 Security Considerations

1. **Input Validation**: All inputs are validated using Pydantic
2. **CORS**: Configured for cross-origin requests
3. **File Upload**: Temporary files are cleaned up after processing
4. **Environment Variables**: Use `.env` for sensitive configurations

## 📄 License

This project is licensed under the MIT License.

## 🎯 Future Enhancements

- [ ] Advanced caching strategies
- [ ] Streaming responses
- [ ] Multi-language support
- [ ] Graph-based retrieval
- [ ] Semantic caching
- [ ] Query rewriting
- [ ] Chain-of-thought prompting
- [ ] Ensemble models
- [ ] Real-time indexing

## 📚 References

- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [RAGAS Framework](https://docs.ragas.io/)
- [MLFlow Documentation](https://mlflow.org/docs/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

**Last Updated**: March 7, 2026  
**Version**: 1.0.0


