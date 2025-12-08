"""
Text2SQL Backend on Google Colab GPU
Run this script in a Colab cell to expose your backend on GPU via ngrok
"""

import subprocess
import time
import os
import sys
import requests
import json

def run_command(cmd, description=""):
    """Run shell command and return output."""
    if description:
        print(f"\n▶️  {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def main():
    print("="*60)
    print("🚀 Text2SQL Backend GPU Setup (Colab)")
    print("="*60)
    
    # 1. Clone from GitHub
    print("\n1️⃣  Cloning from GitHub...")
    repo_url = "https://github.com/Tanishque99/text2sql.git"
    run_command(f"git clone {repo_url} /content/text2sql", "Cloning repository")
    
    # Check folder structure
    print("\n📁 Checking folder structure...")
    content_dir = "/content/text2sql"
    if os.path.exists(content_dir):
        print(f"   Contents of {content_dir}:")
        for item in os.listdir(content_dir):
            print(f"     - {item}")
    
    # Determine backend directory based on structure
    backend_dir = None
    if os.path.exists(os.path.join(content_dir, "backend")):
        backend_dir = os.path.join(content_dir, "backend")
    elif os.path.exists(os.path.join(content_dir, "src")):
        backend_dir = os.path.join(content_dir, "src")
    else:
        # Assume current repo is backend itself
        backend_dir = content_dir
    
    print(f"✓ Using backend directory: {backend_dir}")
    
    # 2. Check GPU
    print("\n2️⃣  Checking GPU...")
    gpu_result = run_command("nvidia-smi", "GPU Status")
    has_gpu = gpu_result is not None
    
    if has_gpu:
        print("✓ GPU detected - will use fine-tuned model")
        recommended_model = "tzaware/codet5p-spider-finetuned"
    else:
        print("❌ No GPU detected")
        print("   IMPORTANT: Go to Runtime > Change runtime type > Select 'GPU' (T4 or L4)")
        print("   The fine-tuned model requires GPU and will crash on CPU")
        print("   Please enable GPU first, then run this script again")
        return
    
    # 3. Verify backend structure
    print("\n3️⃣  Verifying backend structure...")
    required_files = ["main.py", "config.py", "requirements.txt"]
    missing_files = []
    for file in required_files:
        file_path = os.path.join(backend_dir, file)
        if os.path.exists(file_path):
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} (missing)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        print(f"   Backend directory contents:")
        for item in os.listdir(backend_dir):
            print(f"     - {item}")
        return
    
    # 3. Install dependencies FIRST (before checking FAISS)
    print("\n3️⃣  Installing dependencies...")
    run_command("pip install -q fastapi uvicorn pydantic python-dotenv", "Installing FastAPI")
    run_command("pip install -q faiss-cpu sentence-transformers", "Installing FAISS & embeddings")
    run_command("pip install -q requests", "Installing requests")
    run_command("pip install -q pyngrok", "Installing ngrok")
    run_command("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", "Installing PyTorch GPU")
    run_command("pip install -q transformers", "Installing transformers")
    
    # Check for FAISS files
    faiss_dir = os.path.join(backend_dir, "vector_service", "output")
    faiss_index = os.path.join(faiss_dir, "faiss.index")
    metadata_file = os.path.join(faiss_dir, "metadata.json")
    
    if not os.path.exists(faiss_index):
        print(f"\n⚠️  FAISS index not found: {faiss_index}")
        print(f"   Creating placeholder FAISS index for Colab...")
        
        # Create dummy FAISS index to prevent startup crash
        os.makedirs(faiss_dir, exist_ok=True)
        
        # Create a minimal FAISS index
        import faiss
        import numpy as np
        
        # Create a simple 384-dim index (for all-MiniLM-L6-v2)
        index = faiss.IndexFlatL2(384)
        index.add(np.zeros((1, 384), dtype=np.float32))
        faiss.write_index(index, faiss_index)
        
        # Create dummy metadata
        metadata = [{"type": "schema", "text": "Database schema"}]
        import json as json_lib
        with open(metadata_file, 'w') as f:
            json_lib.dump(metadata, f)
        
        print(f"   ✓ Created placeholder FAISS index")
        print(f"   ⚠️  NOTE: The server will start but retrieval may not work properly")
        print(f"   To use real data, upload your vector_service/output/ files to Colab")
    else:
        print(f"   ✓ FAISS index found")
    
    # 4. Setup environment
    print("\n4️⃣  Setting up environment...")
    hf_token = "YOUR_HF_TOKEN"  # Replace with your token
    os.environ['HUGGINGFACE_API_TOKEN'] = hf_token
    
    # Fix main.py to handle Colab paths correctly
    main_file = os.path.join(backend_dir, "main.py")
    with open(main_file, 'r') as f:
        main_content = f.read()
    
    # Replace the path logic to use absolute paths in Colab
    old_path_logic = '''    # Initialize vector retriever
    # Paths are relative to the project root (parent of backend)
    backend_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(backend_dir)
    index_path = os.path.join(project_root, "backend", config.FAISS_INDEX_PATH)
    metadata_path = os.path.join(project_root, "backend", config.METADATA_PATH)'''
    
    new_path_logic = '''    # Initialize vector retriever
    # Handle both local and Colab paths
    backend_dir = os.path.dirname(__file__)
    
    # Check if running in Colab (has /content/text2sql structure)
    if os.path.exists("/content/text2sql/backend/vector_service"):
        index_path = "/content/text2sql/backend/vector_service/output/faiss.index"
        metadata_path = "/content/text2sql/backend/vector_service/output/metadata.json"
    else:
        # Local paths - relative to project root
        project_root = os.path.dirname(backend_dir)
        index_path = os.path.join(project_root, "backend", config.FAISS_INDEX_PATH)
        metadata_path = os.path.join(project_root, "backend", config.METADATA_PATH)
    
    print(f"Using FAISS index: {index_path}")
    print(f"Using metadata: {metadata_path}")'''
    
    main_content = main_content.replace(old_path_logic, new_path_logic)
    
    with open(main_file, 'w') as f:
        f.write(main_content)
    
    print(f"✓ Fixed main.py path logic for Colab")
    
    # Write config.py to set the right model based on GPU availability
    config_file = os.path.join(backend_dir, "config.py")
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    # Update MODEL_NAME to recommended model
    config_content = config_content.replace(
        'MODEL_NAME = "tzaware/codet5p-spider-finetuned"',
        f'MODEL_NAME = "{recommended_model}"'
    )
    
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✓ config.py updated to use model: {recommended_model}")
    
    env_file = os.path.join(backend_dir, ".env")
    with open(env_file, 'w') as f:
        f.write(f"HUGGINGFACE_API_TOKEN={hf_token}\n")
        f.write("USE_HF_INFERENCE_API=True\n")
        f.write("CUSTOM_MODEL_API_URL=\n")
    print(f"✓ .env file created")
    
    # 4b. Fix API endpoint in sql_generator.py to use router
    print("\n4b️⃣  Fixing HuggingFace API endpoint...")
    sql_gen_file = os.path.join(backend_dir, "services", "sql_generator.py")
    if os.path.exists(sql_gen_file):
        with open(sql_gen_file, 'r') as f:
            sql_content = f.read()
        # Replace deprecated api-inference with router
        sql_content = sql_content.replace(
            'f"https://api-inference.huggingface.co/models/{model_name}"',
            'f"https://router.huggingface.co/models/{model_name}"'
        )
        with open(sql_gen_file, 'w') as f:
            f.write(sql_content)
        print(f"✓ Updated sql_generator.py to use router endpoint")
    
    # 5. Setup ngrok
    print("\n5️⃣  Setting up ngrok...")
    ngrok_token = "YOUR_NGROK_TOKEN"  # Replace with your token
  
    
    from pyngrok import ngrok
    ngrok.set_auth_token(ngrok_token)
    print("✓ ngrok configured")
    
    # 6. Start server
    print("\n6️⃣  Starting backend server...")
    os.chdir(backend_dir)
    sys.path.insert(0, backend_dir)
    
    # Start server with BOTH stdout and stderr captured
    proc = subprocess.Popen(
        ['python', '-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000'],
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        text=True,
        bufsize=1
    )
    
    print("   Waiting for server startup (20 seconds max)...")
    server_started = False
    startup_timeout = 20
    
    for i in range(startup_timeout):
        time.sleep(1)
        
        # Check if server is still running
        if proc.poll() is not None:
            print(f"\n❌ Server crashed with exit code {proc.poll()}")
            return
        
        # Check if we see the startup message
        if i >= 15:  # After 15 seconds, assume it's ready
            server_started = True
            break
        
        if i % 5 == 0 and i > 0:
            print(f"   Still starting... ({i}s)")
    
    if server_started:
        print("✓ Backend server started on 0.0.0.0:8000")
    else:
        print("⚠️  Server startup timeout - proceeding anyway")
    
    # Skip the local health check and proceed directly to ngrok
    time.sleep(2)
    
    # 7. Expose via ngrok
    print("\n7️⃣  Exposing via ngrok...")
    try:
        tunnel = ngrok.connect(8000, "http")
        # Extract the public URL from the tunnel object
        public_url = tunnel.public_url if hasattr(tunnel, 'public_url') else str(tunnel).split('"')[1]
        print(f"✅ Public URL: {public_url}")
        print(f"\n📍 API Endpoints:")
        print(f"   - Health: {public_url}/health")
        print(f"   - Docs: {public_url}/docs")
        print(f"   - Generate: POST {public_url}/generate")
        print(f"   - Retrieve: POST {public_url}/retrieve")
    except Exception as e:
        print(f"❌ ngrok connection failed: {e}")
        return
    
    # 8. Test API
    print("\n8️⃣  Testing API...")
    print("   Waiting 5 seconds before testing...")
    time.sleep(5)
    
    try:
        health_url = f"{public_url}/health"
        print(f"   Testing: {health_url}")
        health_resp = requests.get(health_url, timeout=15)
        if health_resp.status_code == 200:
            print("✓ Health check passed!")
            print(json.dumps(health_resp.json(), indent=2))
        else:
            print(f"⚠️  Health check returned {health_resp.status_code}")
            print(f"Response: {health_resp.text[:200]}")
    except Exception as e:
        print(f"⚠️  Health check error: {e}")
        print("   The backend may still be initializing...")
    
    # 9. Summary
    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("="*60)
    print(f"\n🔗 Public URL (copy for frontend/local machine):")
    print(f"   {public_url}")
    print(f"\n📝 Next steps:")
    print(f"   1. Update frontend to call: {public_url}")
    print(f"   2. Or set CUSTOM_MODEL_API_URL={public_url} in local .env")
    print(f"   3. Keep this Colab cell running to maintain the server")
    print(f"\n⚠️  Server initialization:")
    print(f"   - FAISS and embeddings load on first request")
    print(f"   - First query may take 30-60 seconds")
    print(f"   - Subsequent queries are faster")
    print(f"\n⏰ Monitoring server status:")
    
    # Keep server alive with monitoring
    import datetime
    start_time = time.time()
    try:
        while True:
            elapsed = int(time.time() - start_time)
            if proc.poll() is not None:
                print(f"\n⚠️  [{elapsed}s] Server process exited with code {proc.poll()}")
                break
            
            # Every 60 seconds, print status
            if elapsed % 60 == 0 and elapsed > 0:
                try:
                    resp = requests.get("http://localhost:8000/health", timeout=5)
                    status = "✓ OK" if resp.status_code == 200 else f"✗ {resp.status_code}"
                    print(f"   [{elapsed}s] Server health: {status}")
                except:
                    print(f"   [{elapsed}s] Server health: ⚠️  No response")
            
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()
        print("✓ Server stopped")


if __name__ == "__main__":
    main()
