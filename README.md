# rag_metrics
Observability for RAG
Metrics in your RAG evaluation (like MAP, nDCG, Recall, Precision) are computed based on comparing the retriever’s output (run) against the ground truth (qrels)

{
  "q1": {"doc88": 1, "doc87": 0, "doc3": 0},
  "q2": {"doc77": 1, "doc78": 0, "doc3": 0}
}
1 → relevant, 0 → not relevant

Imagine you’re fishing:
Recall = Did you catch all the fish in the pond?
Precision = Of the fish you caught, how many are edible?
MAP = Average “goodness” of fish caught in order.
nDCG = Are the best fish caught first?


## Setup Instructions

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