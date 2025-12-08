# Text2SQL Backend on Google Colab GPU

Quick guide to run your Text2SQL backend on Google Colab's free GPU with a public URL.

## Prerequisites

1. **Google Colab** account (free): https://colab.research.google.com
2. **HuggingFace API token**: https://huggingface.co/settings/tokens
   - Create a token and copy it
3. **ngrok account** (free): https://dashboard.ngrok.com/auth/your-authtoken
   - Get your auth token from the link above

## Setup Steps

### Step 1: Create a New Colab Notebook

1. Go to https://colab.research.google.com
2. Click **File → New Notebook**
3. Rename it (e.g., "Text2SQL Backend GPU")

### Step 2: Upload Your Code

**Option A: Git Clone** (if your repo is on GitHub)
```python
!git clone https://github.com/yourusername/text2sql-master.git /content/text2sql
```

**Option B: Manual Upload** (if not on GitHub)
1. Click **Files** icon on the left (folder icon)
2. Click **Upload** (file icon)
3. Select your `text2sql-master` folder (or individual files)
4. Upload to `/content`

### Step 3: Install Setup Script

Copy this into a Colab cell and run:

```python
# Download setup script
!wget -q https://raw.githubusercontent.com/yourusername/text2sql-master/main/backend/colab_setup.py -O colab_setup.py

# Or paste the content directly from backend/colab_setup.py
%paste
```

Then run the setup:

```python
exec(open('colab_setup.py').read())
```

### Step 4: Configure Tokens

Before running, edit the script variables:

```python
# Line 37: Replace with your HuggingFace token
hf_token = "hf_YOUR_TOKEN_HERE"

# Line 40: Replace with your ngrok token
ngrok_token = "YOUR_NGROK_AUTH_TOKEN"
```

### Step 5: Run Setup

Run the script cell. You should see:

```
🚀 Text2SQL Backend GPU Setup (Colab)
============================================================

1️⃣  Checking GPU...
✓ GPU detected

2️⃣  Cloning repository...
✓ Repository cloned

3️⃣  Installing dependencies...
✓ All packages installed

4️⃣  Setting up environment...
✓ .env file created

5️⃣  Setting up ngrok...
✓ ngrok configured

6️⃣  Starting backend server...
✓ Backend server started

7️⃣  Exposing via ngrok...
✅ Public URL: https://abcd-1234-ef.ngrok.io

📍 API Endpoints:
   - Health: https://abcd-1234-ef.ngrok.io/health
   - Docs: https://abcd-1234-ef.ngrok.io/docs
   - Generate: POST https://abcd-1234-ef.ngrok.io/generate
   - Retrieve: POST https://abcd-1234-ef.ngrok.io/retrieve

8️⃣  Testing API...
✓ Health check passed

✅ Setup complete!
```

### Step 6: Use the Public URL

#### Option A: From Your Local Machine

Update your frontend or local backend `.env`:

```bash
CUSTOM_MODEL_API_URL=https://abcd-1234-ef.ngrok.io
```

Then restart your local backend.

#### Option B: Direct API Calls

```python
import requests

url = "https://abcd-1234-ef.ngrok.io/generate"
payload = {
    "question": "How many students are there?",
    "db_id": None,
    "top_k": 5
}

response = requests.post(url, json=payload)
print(response.json())
```

### Step 7: Keep Server Running

**Keep the setup cell running!** The ngrok tunnel expires when:
- You stop the cell
- Colab disconnects (15+ min inactivity)
- Session ends

To keep it alive, you can add this to keep the cell running:

```python
import time
print("✅ Backend running on GPU!")
print(f"Public URL: {public_url}")
print("Keep this cell running to maintain the server.\n")

try:
    while True:
        time.sleep(60)
        # Optional: Add monitoring here
except KeyboardInterrupt:
    print("\nShutting down...")
```

## Troubleshooting

### "Unable to read file" error
- This usually means the `.ipynb` file format was corrupted
- **Solution**: Use the `colab_setup.py` script instead (simpler, more reliable)

### ngrok connection fails
- Check your ngrok auth token: https://dashboard.ngrok.com/auth/your-authtoken
- Make sure you copied it correctly (no extra spaces)

### API returns 503 (Model Loading)
- The HuggingFace model is initializing (can take 1-2 min on first call)
- Wait and retry

### "HUGGINGFACE_API_TOKEN must be set"
- Verify your HuggingFace token is correct and has proper permissions
- Make sure you replaced the placeholder in the script

### Colab disconnects
- Set Colab to stay connected: **Tools → Settings → Notebook settings → Omit code cell output**
- Keep your browser tab active
- Consider Colab Pro ($10/mo) for longer sessions and better GPU

## Performance Tips

- **GPU**: Default T4 is free; Colab Pro adds A100 (10x faster)
- **FAISS**: Runs on CPU but is fast for vector search
- **Model**: Uses HuggingFace Inference API (serverless) — no GPU needed on Colab
- **Bottleneck**: Usually HuggingFace API response time, not Colab

## Cleanup

To stop and cleanup:

```python
ngrok.disconnect(public_url)
proc.terminate()
print("✓ Server stopped")
```

## Example Integration

### From Your Frontend (React/Vue)

```javascript
const API_URL = "https://your-ngrok-url";

async function generateSQL(question) {
  const response = await fetch(`${API_URL}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: question,
      db_id: null,
      top_k: 5
    })
  });
  
  const data = await response.json();
  return data.generated_sql;
}
```

### From Your Flask Frontend

```python
import requests

API_URL = "https://your-ngrok-url"

@app.route("/query", methods=["POST"])
def query():
    question = request.json.get("question")
    response = requests.post(
        f"{API_URL}/generate",
        json={"question": question, "top_k": 5}
    )
    return response.json()
```

## Next Steps

1. ✅ Backend running on GPU
2. 🔗 Public URL exposed via ngrok
3. 📱 Update frontend to call the public URL
4. 🧪 Test end-to-end flow

Happy querying! 🚀
