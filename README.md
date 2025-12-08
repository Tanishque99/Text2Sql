# Text2SQL

Convert natural language questions to SQL queries using CodeT5+ and RAG (Retrieval-Augmented Generation).

## 🚀 Features

- **Natural Language Processing**: Convert plain English questions to SQL queries
- **GPU-Powered Inference**: Uses Google Colab with T4 GPU for fast model inference
- **Semantic Search**: FAISS vector store with 9,535+ Spider dataset examples
- **RAG Architecture**: Retrieval-Augmented Generation for context-aware SQL
- **Side-by-Side View**: Modern UI showing question, SQL, schema, and similar examples
- **FastAPI Backend**: High-performance API with automatic documentation
- **Flask Frontend**: Clean, responsive web interface

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

## 📋 Prerequisites

- Python 3.10+ (avoid 3.14 due to tokenizer compatibility issues)
- Google Colab account (free tier with T4 GPU)
- ngrok account (free tier)
- Spider dataset (download from [Yale Spider](https://yale-lily.github.io/spider))
- 4GB+ RAM for FAISS index

## 🛠️ Setup

### Option 1: Quick Start with Colab GPU (Recommended)

This is the fastest way to get started - uses Google Colab's free GPU for model inference.

#### 1. Clone the Repository

```bash
git clone https://github.com/Tanishque99/Text2Sql.git
cd Text2Sql
```

#### 2. Setup Local Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Configure Environment Variables

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

#### 4. Setup Colab GPU Backend

**Option A: Run colab_setup.py in Colab**
1. Open Google Colab: https://colab.research.google.com/
2. Upload `backend/colab_setup.py` or copy its content
3. Update the HuggingFace token in the script
4. Run the cell - it will:
   - Install dependencies
   - Start backend on GPU
   - Expose via ngrok
   - Display public URL

**Option B: Use the Jupyter Notebook**
1. Upload `backend/colab/text2sql_backend_gpu.ipynb` to Colab
2. Get ngrok auth token from https://dashboard.ngrok.com/get-started/your-authtoken
3. Update tokens in the notebook
4. Run all cells
5. Copy the ngrok URL (e.g., `https://xxx.ngrok-free.app`)

#### 5. Update Local Backend Configuration

Update your `.env` file with the Colab ngrok URL:
```bash
CUSTOM_MODEL_API_URL=https://your-actual-ngrok-url.ngrok-free.app
```

#### 6. Setup Frontend

```bash
cd ../frontend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 7. Run the Application

**Terminal 1: Start Local Backend**
```bash
cd backend
source venv/bin/activate

# macOS: Set threading environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Start Frontend**
```bash
cd frontend
source venv/bin/activate
python app.py
```

Frontend will be available at: http://localhost:5001

---

### Option 2: Full Local Setup (Advanced)

For users who want to process the Spider dataset from scratch and run everything locally.

#### 1. Clone the Repository

```bash
git clone https://github.com/Tanishque99/Text2Sql.git
cd Text2Sql
```

#### 2. Download Spider Dataset

Download the Spider dataset and extract it:
```bash
# Download from https://yale-lily.github.io/spider
# Extract to your preferred location, e.g., ~/Downloads/spider_data
```

#### 3. Setup Vector Service

```bash
cd backend/vector_service

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Process Spider dataset
python preprocess_spider.py \
  --spider-dir /path/to/spider/data/directory \
  --database-dir /path/to/spider/data/directory/database

# Generate FAISS index (this may take 5-10 minutes)
python ingest.py --input output/spider_processed.json
```

**Note:** The FAISS index is already included in `vector_service/output/` with 9,535 vectors, so you can skip this step unless you want to rebuild it.

#### 4. Setup Backend

```bash
cd ../  # Move to backend directory

# Use the same virtual environment
source vector_service/venv/bin/activate

# Install backend dependencies (if not already installed)
pip install -r requirements.txt
```

#### 5. Setup Frontend

```bash
cd ../frontend

# Create virtual environment for frontend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 6. Run the Application

**Terminal 1: Start Backend Server**
```bash
cd backend
source vector_service/venv/bin/activate

# macOS: Set threading environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

PYTHONPATH=. python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be available at: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

**Terminal 2: Start Frontend Server**
```bash
cd frontend
source venv/bin/activate
python app.py
```

Frontend will be available at: http://localhost:5001

---

## 💡 Usage

1. Open your browser and navigate to http://localhost:5001
2. Enter a natural language question (e.g., "How many singers do we have?")
3. Click "Generate SQL"
4. View the results in the side-by-side comparison:
   - **Left Panel**: Your question and generated SQL
   - **Right Panel**: Database schema and similar examples

## 📁 Project Structure

```
Text2Sql/
├── backend/
│   ├── main.py                      # FastAPI application
│   ├── config.py                    # Configuration settings
│   ├── requirements.txt             # Backend dependencies
│   ├── colab_setup.py              # Automated Colab setup script
│   ├── .env.example                # Environment variables template
│   ├── models/
│   │   └── schemas.py              # Pydantic models
│   ├── services/
│   │   ├── retriever.py            # FAISS vector search
│   │   └── sql_generator.py        # SQL generation logic
│   ├── colab/
│   │   └── text2sql_backend_gpu.ipynb  # Colab GPU notebook
│   └── vector_service/
│       ├── preprocess_spider.py    # Spider dataset processor
│       ├── ingest.py               # FAISS index generator
│       ├── requirements.txt        # Vector service dependencies
│       └── output/                 # Generated files (9,535 vectors)
│           ├── spider_processed.json
│           ├── faiss.index
│           ├── embeddings.npy
│           └── metadata.json
└── frontend/
    ├── app.py                      # Flask application
    ├── requirements.txt            # Frontend dependencies
    ├── templates/
    │   └── index.html              # Main UI template
    └── static/
        ├── css/
        │   └── style.css           # Styling
        └── js/
            └── main.js             # Frontend logic
```

## 🔧 Configuration

### Backend Configuration (`backend/config.py`)

- `FAISS_INDEX_PATH`: Path to FAISS index file
- `METADATA_PATH`: Path to metadata JSON file
- `EMBEDDING_MODEL`: Sentence transformer model name
- `MODEL_NAME`: CodeT5+ model for SQL generation
- `TOP_K`: Number of similar examples to retrieve

### Environment Variables

Create a `.env` file in `backend/` directory:
```bash
HUGGINGFACE_API_TOKEN=your_hf_token_here
CUSTOM_MODEL_API_URL=https://your-ngrok-url.ngrok-free.app
USE_HF_INFERENCE_API=False
```

## API Endpoints

### Local Backend (localhost:8000)

- `GET /health` - Health check
- `POST /generate` - Generate SQL from question
- `POST /retrieve` - Retrieve similar examples

**Example:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many singers do we have?"}'
```

## 🧪 Testing

### Test Backend API

```bash
cd backend
source venv/bin/activate
python test_api.py
```

### Test Vector Service

```bash
cd backend
source venv/bin/activate
python debug_payload.py
```

## 📊 Dataset Information

- **Databases**: 166 different database schemas
- **Total Entries**: 9,535 (876 schemas + 8,659 examples)
- **Source**: Spider dataset from Yale NLP
- **Format**: JSON with schema details, foreign keys, primary keys, and SQL examples
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **SQL Generation Model**: tzaware/codet5p-spider-finetuned

## 🐛 Troubleshooting

### Segmentation Fault on macOS
Set threading environment variables before running:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

### FAISS Index Not Found
The index is already included in `backend/vector_service/output/`. To rebuild:
```bash
cd backend/vector_service
source venv/bin/activate
python ingest.py --input output/spider_processed.json
```

### Colab Server Crashes
- Ensure FAISS index exists or will be created
- Check ngrok auth token is valid
- Verify HuggingFace token has read permissions

### Port Already in Use
```bash
# Backend (change port)
uvicorn main:app --port 8001

# Frontend (edit app.py to change port)
```

### Module Import Errors
```bash
# Make sure PYTHONPATH is set for backend
cd backend
PYTHONPATH=. python -m uvicorn main:app
```

### Model Loading Errors
- Use Python 3.10-3.12 (avoid 3.14)
- Ensure Colab has GPU runtime enabled
- Check HuggingFace token is valid

## Performance

- **FAISS Retrieval**: ~50-100ms for 5 similar examples
- **Model Inference**: ~1-3s on Colab T4 GPU
- **Total Response Time**: ~1.5-4s end-to-end
- **Vector Store Size**: ~50MB (9,535 vectors)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Spider Dataset**: Yale NLP Group
- **FAISS**: Facebook AI Research
- **Sentence Transformers**: UKPLab
- **CodeT5+**: Salesforce Research
- **ngrok**: Tunneling service for Colab access

## 📧 Contact

For questions or issues, please open an issue on GitHub.
