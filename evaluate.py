import json
from ranx import Qrels, Run, evaluate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Paths
FAISS_INDEX_PATH = "faiss_index"
QUERIES_JSON_PATH = "queries.json"
QRELS_JSON_PATH = "qrels.json"

def main():
    # 1. Load retriever
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 2. Load queries
    with open(QUERIES_JSON_PATH, "r") as f:
        queries = json.load(f)

    # 3. Build Run (retriever outputs for evaluation)
    run = {}
    for q in queries:
        query_id = q["id"]
        query_text = q["text"]
        docs = retriever.invoke(query_text)
        run[query_id] = {doc.metadata["id"]: 1.0 for doc in docs}

    run = Run(run)

    # 4. Load qrels (ground truth relevance)
    qrels = Qrels.from_file(QRELS_JSON_PATH)

    # 5. Evaluate with common IR metrics , Compares the retrieved documents (run) against ground truth (qrels).
    results = evaluate(qrels, run, metrics=["map@5", "ndcg@5", "recall@5", "precision@5"])

    print("Evaluation Results:")
    print(results)

if __name__ == "__main__":
    main()
