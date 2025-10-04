# rag_metrics
Observability for RAG

## 🚀 Setup Instructions

### 1. Clone the project
```bash
git clone https://github.com/sunnyaws1984/rag_metrics.git
cd rag_metrics

2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Mac/Linux
source .venv/Scripts/activate   # GIT BASH

3. Install langchain and RANX

uv pip install -r requirements.txt (Speed up the installations)

4. Commands to run:
python create_index.py
python evaluate.py