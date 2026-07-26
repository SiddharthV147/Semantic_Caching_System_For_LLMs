# LMS Semantic Cache 

A high-performance, context-aware semantic caching system for Learning Management Systems (LMS) designed to dramatically reduce AI API costs and serve instant responses to rephrased student questions.

---

## System Architecture

```
                      ┌─────────────────────────────────┐
                      │     Student Asks a Question     │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   Convert Question to Vector  │
                       │   (BAAI/bge-large-en-v1.5)    │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Similarity Search in Milvus   │
                       └───────────────┬───────────────┘
                                       │
                       ┌───────────────┴───────────────┐
                       │                               │
            Cosine Similarity ≥ 0.92?       Cosine Similarity < 0.92
                  [ CACHE HIT ]                   [ CACHE MISS ]
                       │                               │
                       ▼                               ▼
            ┌─────────────────────┐         ┌─────────────────────┐
            │  Fetch Answer from  │         │ Call LLM API        │
            │        Redis        │         │ (Qwen2.5 via HF)    │
            └──────────┬──────────┘         └──────────┬──────────┘
                       │                               │
                       │                    ┌──────────┴──────────┐
                       │                    │ Save Embedding to   │
                       │                    │ Milvus & Redis      │
                       │                    └──────────┬──────────┘
                       │                               │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │   Return Answer to Student  │
                        └─────────────────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Vector DB** | [Milvus](https://milvus.io/) | Stores question embeddings & performs fast similarity searches |
| **Cache Store** | [Redis](https://redis.io/) | Key-value store for direct answer retrieval |
| **Embedding Model** | `BAAI/bge-large-en-v1.5` | Converts text questions into numerical vectors |
| **LLM Inference** | HuggingFace Inference API | Generates answers on cache miss using `Qwen2.5-7B-Instruct` |
| **API Framework** | FastAPI | Asynchronous Python REST API framework |
| **Orchestration** | Docker Compose | Local deployment of Redis & Milvus containers |

---

## Project Structure

```text
lms-semantic-cache/
├── config/                  # Global application settings & thresholds
├── src/
│   ├── api/                 # FastAPI routes and request/response models
│   ├── cache/               # Cache read/write orchestration logic
│   ├── database/            # Connection setups for Milvus & Redis
│   ├── embeddings/          # Vector conversion module (HuggingFace/SentenceTransformers)
│   ├── knowledge/           # Course knowledge base management
│   ├── llm/                 # LLM client & prompt templates
│   └── orchestrator.py      # Core workflow pipeline (Cache → KB → LLM)
├── tests/                   # Unit, integration, and smoke tests
├── demo_queries.json        # Benchmark dataset (60 queries across 3 courses)
├── demo_cache_performance.py# Performance benchmark script
├── main.py                  # API server startup entrypoint
└── docker-compose.yml       # Infrastructure orchestration
```

---

## Getting Started

### Prerequisites

* **Python 3.11+**
* **Docker & Docker Compose**
* **HuggingFace API Token** (Free token available at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

### 1. Clone & Set Up Environment

```bash
git clone <your-repo-url>
cd lms-semantic-cache

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` to include your HuggingFace Token:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
SIMILARITY_THRESHOLD=0.92
EMBEDDING_MODEL_NAME=BAAI/bge-large-en-v1.5
MILVUS_URI=http://localhost:19530
REDIS_URL=redis://localhost:6379/0
```

### 3. Start Infrastructure

Boot up local instances of Milvus and Redis using Docker Compose:

```bash
docker compose up -d
```

### 4. Initialize Database & Run Server

```bash
# Initialize vector collections for courses
python -m src.database.db_setup CS101 MATH202 ENG301

# Start the FastAPI server
python main.py
```

The application will be running at `http://localhost:8000`.  
Explore the interactive API documentation at **`http://localhost:8000/docs`**.

---

## API Reference

### Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check health status of API, Redis, and Milvus |
| `GET` | `/api/v1/courses` | List all active/registered course caches |
| `POST` | `/api/v1/courses` | Register a new course environment |
| `POST` | `/api/v1/query` | Submit a student query (main endpoint) |
| `DELETE` | `/api/v1/cache/{course_tag}` | Flush cached data for a specific course |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is gradient descent?",
    "course_tag": "CS101"
  }'
```

---

## Performance & Benchmarks

Benchmarked using `demo_cache_performance.py` over a dataset of 60 queries across 20 concept groups:

* **Cache Hit Rate Target:** ~65%
* **Cache Hit Latency:** ~400–600 ms (Vector embedding generation + Redis lookup)
* **Cache Miss Latency:** ~5,000–8,000 ms (Vector generation + Remote LLM API invocation)
* **Performance Gain:** **~10x to 15x speedup** on cache hits
