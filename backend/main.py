"""
Main FastAPI application for Text2SQL backend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from models.schemas import QueryRequest, QueryResponse, HealthResponse, ErrorResponse, PromptRequest
from services.retriever import VectorRetriever
from services.sql_generator import SQLGenerator
import config


# Global variables for services
retriever = None
generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global retriever, generator
    
    print("🚀 Starting Text2SQL Backend...")
    
    # Initialize vector retriever
    # Paths are relative to the project root (parent of backend)
    backend_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(backend_dir)
    index_path = os.path.join(project_root, "backend", config.FAISS_INDEX_PATH)
    metadata_path = os.path.join(project_root, "backend", config.METADATA_PATH)
    
    retriever = VectorRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model=config.EMBEDDING_MODEL
    )
    
    # Initialize SQL generator
    generator = SQLGenerator(
        model_name=config.MODEL_NAME,
        max_length=config.MODEL_MAX_LENGTH,
        use_api=config.USE_HF_INFERENCE_API,
        api_token=config.HUGGINGFACE_API_TOKEN,
        custom_api_url=config.CUSTOM_MODEL_API_URL
    )
    
    print("✓ All services initialized successfully!")
    
    yield
    
    # Cleanup (if needed)
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Text2SQL API",
    description="Convert natural language queries to SQL using CodeT5+ and RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Text2SQL API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=generator is not None,
        index_loaded=retriever is not None,
        total_vectors=retriever.index.ntotal if retriever else 0
    )


@app.post("/generate", response_model=QueryResponse, tags=["SQL Generation"])
async def generate_sql(request: QueryRequest):
    """
    Generate SQL query from natural language question.
    
    - **question**: Natural language question
    - **db_id**: Optional database identifier for filtering context
    - **top_k**: Number of context entries to retrieve (default: 5)
    
    Note: FAISS retrieval is always enabled for better SQL generation.
    """
    try:
        print(f"Received request: question='{request.question}', db_id={request.db_id}, top_k={request.top_k}")
        
        # Always retrieve context from FAISS
        print(f"Retrieving context with top_k={request.top_k}")
        if request.db_id:
            # Retrieve context for specific database
            context = retriever.retrieve_by_db(
                query=request.question,
                db_id=request.db_id,
                top_k=request.top_k
            )
        else:
            # General retrieval
            context = retriever.retrieve(
                query=request.question,
                top_k=request.top_k
            )
        
        print(f"Retrieved {len(context)} context entries")
        retrieved_context = context
        
        # Generate SQL
        print("Generating SQL...")
        sql_query = generator.generate_sql(
            question=request.question,
            context=context
        )
        
        print(f"Generated SQL: {sql_query}")
        
        return QueryResponse(
            question=request.question,
            generated_sql=sql_query,
            retrieved_context=retrieved_context,
            db_id=request.db_id
        )
    
    except Exception as e:
        print(f"ERROR in generate_sql endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_from_prompt", tags=["SQL Generation"])
async def generate_from_prompt(request: PromptRequest):
    """
    Generate SQL from a pre-built prompt (for GPU backend inference).
    This endpoint receives the full prompt with schema and examples already included.
    
    - **prompt**: Complete prompt with schema, examples, and question
    - **max_new_tokens**: Maximum tokens to generate
    - **temperature**: Sampling temperature
    - **do_sample**: Whether to use sampling
    """
    try:
        print(f"Received prompt-based request (length: {len(request.prompt)} chars)")
        
        # Generate directly from prompt without retrieval
        # Use the generator's local model if available
        if hasattr(generator, 'use_api') and not generator.use_api:
            # Use local model
            sql_query = generator._generate_with_local_model(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                num_beams=1
            )
        else:
            # Fallback: shouldn't happen on GPU backend, but handle gracefully
            print("WARNING: GPU backend called with use_api=True, this shouldn't happen")
            sql_query = generator.generate_sql(
                question=request.prompt,
                context=None,
                max_new_tokens=request.max_new_tokens
            )
        
        # Clean the SQL
        sql_query = generator._clean_generated_sql(sql_query, "")
        
        print(f"Generated SQL: {sql_query}")
        
        return {
            "generated_sql": sql_query,
            "prompt_length": len(request.prompt)
        }
    
    except Exception as e:
        print(f"ERROR in generate_from_prompt endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", tags=["Retrieval"])
async def retrieve_context(question: str, db_id: str = None, top_k: int = 5):
    """
    Retrieve relevant context for a question (for debugging/testing).
    
    - **question**: Natural language question
    - **db_id**: Optional database identifier
    - **top_k**: Number of results to retrieve
    """
    try:
        if db_id:
            context = retriever.retrieve_by_db(
                query=question,
                db_id=db_id,
                top_k=top_k
            )
        else:
            context = retriever.retrieve(
                query=question,
                top_k=top_k
            )
        
        return {"context": context}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG
    )
