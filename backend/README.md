# Text2SQL Backend

FastAPI backend for converting natural language to SQL queries using CodeT5+ and RAG (Retrieval-Augmented Generation).

## Architecture

**Current Setup:**
- **Frontend**: Flask (localhost:5001) - Web UI
- **Local Backend**: FastAPI (localhost:8000) - FAISS retrieval + request routing
- **Colab Backend**: FastAPI via ngrok - GPU model inference
- **Vector Store**: FAISS with 9,535 Spider dataset entries
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Model**: tzaware/codet5p-spider-finetuned (CodeT5+ fine-tuned on Spider)
- **Connection**: ngrok tunnel from Colab GPU to local backend

**Data Flow:**
```
User → Frontend (Flask :5001)
  ↓
Local Backend (FastAPI :8000)
  ├─→ FAISS (vector retrieval)
  └─→ Colab Backend (ngrok) → GPU Model Inference
```

## Prerequisites

- Python 3.10+ (avoid 3.14 due to tokenizer compatibility issues)
- Google Colab account (free tier with T4 GPU)
- ngrok account (free tier)

## Setup

### 1. Create virtual environment
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. FAISS Vector Store Setup

The FAISS index is already built with 9,535 vectors from Spider dataset:
- Location: `vector_service/output/faiss.index`
- Metadata: `vector_service/output/metadata.json`
- Embeddings: `vector_service/output/embeddings.npy`

**To rebuild (optional):**
```bash
cd vector_service
python preprocess_spider.py  # Process Spider dataset
python ingest.py             # Generate FAISS index
```

### 4. Set up Colab GPU Backend

Use the provided `colab_setup.py` script in Google Colab:

**Option A: Run in Colab Notebook**
```python
# In a Colab cell, paste the entire colab_setup.py content and run
# It will:
# 1. Clone the repo
# 2. Install dependencies
# 3. Start backend on GPU
# 4. Expose via ngrok
# 5. Display public URL
```

**Option B: Manual Colab Setup**
1. Upload `colab/text2sql_backend_gpu.ipynb` to Colab
2. Get ngrok auth token from https://dashboard.ngrok.com/get-started/your-authtoken
3. Run all cells
4. Copy the ngrok URL (e.g., `https://xxx.ngrok-free.app`)

### 5. Configure Environment Variables

Create `.env` file in `backend/` directory:
```bash
HUGGINGFACE_API_TOKEN=your_token_here
CUSTOM_MODEL_API_URL=https://your-ngrok-url.ngrok-free.app
USE_HF_INFERENCE_API=False
```

**Get HuggingFace Token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a token with read access
3. Copy and paste into `.env`

## Running the API

### Start the backend server

**Prerequisites (macOS):**
Set these environment variables to prevent threading issues with the model:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

**Start the server:**
```bash
cd backend
source .venv/bin/activate  # or: source venv/bin/activate
PYTHONPATH=. vector_service/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**Important**: Keep the Colab cell running while using the backend!

## Running Frontend (Optional)

The frontend provides a web UI for the backend:

```bash
cd frontend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Access at: http://localhost:5001

## API Endpoints

### `POST /generate`
Generate SQL from natural language question.

**Request:**
```json
{
  "question": "how many cars with model ford?",
  "db_id": null,
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "how many cars with model ford?",
  "generated_sql": "SELECT count(*) FROM car_names WHERE Model = 'ford'",
  "retrieved_context": [
    {
      "type": "schema",
      "text": "Database: car_1\nTable: car_names\n...",
      "distance": 0.234
    }
  ],
  "db_id": null
}
```

**Parameters:**
- `question`: Natural language query (required)
- `db_id`: Database identifier (optional)
- `top_k`: Number of context entries to retrieve (default: 5)

### `POST /retrieve`
Retrieve similar examples without generating SQL (for debugging).

**Request:**
```json
{
  "question": "Show all singers from USA",
  "top_k": 3
}
```

### `GET /health`
Health check endpoint. Returns backend and retriever status.

### `GET /`
Root endpoint with welcome message.

## Testing

### Test SQL generation
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "how many cars with model ford?", "top_k": 5}'
```

### Test with specific database
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the names of all artists?", "db_id": "concert_singer", "top_k": 3}'
```

### Test retrieval only
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "Show all singers", "top_k": 5}'
```

### Test health endpoint
```bash
curl http://localhost:8000/health
```

## Documentation

Interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Troubleshooting

### Segmentation Fault on macOS
**Issue**: Backend crashes with segmentation fault when loading model locally.
**Solution**: The fine-tuned model is too large for CPU. Use Colab GPU backend instead.
Set environment variables:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

### "Empty reply from server" or 500 errors
- **Check Colab**: Verify Colab cell is still running
- **Check ngrok URL**: Verify `CUSTOM_MODEL_API_URL` in `.env` matches Colab ngrok URL
- **Test Colab health**: `curl https://your-ngrok-url.ngrok-free.app/health`
- **Check .env settings**:
  - `USE_HF_INFERENCE_API=False`
  - `CUSTOM_MODEL_API_URL=<your-ngrok-url>`

### "410 Gone" or deprecated API errors
**Issue**: HuggingFace deprecated `api-inference.huggingface.co` endpoint.
**Solution**: Use Colab GPU backend (`USE_HF_INFERENCE_API=False`) instead of direct API calls.

### FAISS index not found
- Verify file exists: `backend/vector_service/output/faiss.index`
- If missing, included in repo already (9,535 vectors)
- To rebuild: `cd vector_service && python ingest.py`

### Model not found on HuggingFace
**Issue**: `tzaware/codet5p-spider-finetuned` returns 404.
**Solution**: This is a custom fine-tuned model. Use local model loading in Colab (not HuggingFace Inference API).

### Frontend can't connect to backend
- Verify backend is running on `localhost:8000`
- Check `BACKEND_URL` in `frontend/app.py` is set to `http://localhost:8000`
- Test backend: `curl http://localhost:8000/health`

## Project Structure

```
backend/
├── main.py                 # FastAPI application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in git)
├── colab_setup.py         # Colab GPU setup script
├── Dockerfile             # Docker configuration (optional)
├── test_api.py            # API testing script
├── debug_payload.py       # Debug utility
├── evaluate_spider.py     # Spider benchmark evaluation
├── models/
│   └── schemas.py        # Pydantic request/response models
├── services/
│   ├── retriever.py      # FAISS retrieval service
│   └── sql_generator.py  # CodeT5+ inference service
├── colab/
│   ├── text2sql_backend_gpu.ipynb  # Colab notebook
│   └── text2sql_fine_tuning.ipynb  # Fine-tuning notebook
└── vector_service/
    ├── preprocess_spider.py        # Spider dataset preprocessing
    ├── ingest.py                   # FAISS index generation
    ├── output/
    │   ├── faiss.index            # FAISS index (9,535 vectors)
    │   ├── metadata.json          # Vector metadata
    │   ├── embeddings.npy         # Raw embeddings
    │   └── spider_processed.json  # Processed Spider data
    └── spider_data/               # Original Spider dataset
        ├── train_spider.json
        ├── dev.json
        ├── tables.json
        └── database/              # SQLite databases
```

## Files Description

- **colab_setup.py**: Automated Colab deployment script (clones repo, installs deps, starts server, exposes via ngrok)
- **config.py**: Centralized configuration (model name, API tokens, paths)
- **main.py**: FastAPI app with lifespan management for services
- **services/retriever.py**: FAISS vector search for schema/example retrieval
- **services/sql_generator.py**: SQL generation via Colab GPU or HuggingFace API
- **vector_service/output/**: Pre-built FAISS index with 9,535 Spider vectors

## Notes

- **Keep Colab running**: The Colab cell must stay active for the backend to work
- **ngrok URLs**: Free tier URLs expire after inactivity (~2 hours)
- **FAISS index**: Contains 9,535 entries from Spider dataset
- **GPU**: Model uses T4 GPU on Colab free tier
- **Python version**: 3.10-3.13 recommended (avoid 3.14)
- **Model**: Fine-tuned CodeT5+ (tzaware/codet5p-spider-finetuned) optimized for SQL generation
- **Threading**: macOS users must set `OMP_NUM_THREADS=1` to prevent segmentation faults

## Performance

- **First query**: ~30-60 seconds (loads model + embeddings)
- **Subsequent queries**: ~2-5 seconds
- **Vector retrieval**: ~100ms (9,535 vectors)
- **SQL generation**: ~2-3 seconds on T4 GPU

## License

MIT License - See LICENSE file for details
