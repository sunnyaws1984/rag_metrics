import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# Paths (all inside rag_metrics)
PDF_PATH = "sample.pdf"
FAISS_INDEX_PATH = "faiss_index"   # save index at project root
DOCS_JSON_PATH = "docs.json"

def main():
    # 1. Load PDF
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    # 2. Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = splitter.split_documents(pages)
    #print("Documents",documents)

    # 3. Add unique IDs to each chunk
    documents_with_ids = []
    docs_json_data = []
    for i, doc in enumerate(documents):
        doc_id = f"doc{i+1}"
        new_doc = Document(page_content=doc.page_content, metadata={"id": doc_id})
        #print ("new_doc",new_doc)
        documents_with_ids.append(new_doc)

        docs_json_data.append({
            "id": doc_id,
            "contents": doc.page_content
        })

    # 4. Save docs.json
    with open(DOCS_JSON_PATH, "w") as f:
        json.dump(docs_json_data, f, indent=2)

    # 5. Create FAISS index using HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index = FAISS.from_documents(
        documents_with_ids,
        embeddings
    )

    # 6. Save FAISS index
    faiss_index.save_local(FAISS_INDEX_PATH)

    print(f"Created FAISS index at {FAISS_INDEX_PATH}")
    print(f"Created docs.json at {DOCS_JSON_PATH} with {len(docs_json_data)} chunks")

if __name__ == "__main__":
    main()
