import json
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import (Distance,SparseIndexParams,SparseVectorParams,VectorParams)

from src.config import (EMBEDDING_MODEL,OLLAMA_BASE_URL,PROCESSED_DATA_DIR,QDRANT_COLLECTION,QDRANT_URL)

CHUNKS_DATA_DIR = PROCESSED_DATA_DIR.parent / "chunks"
EMBEDDING_DIM = 768 


def _make_deterministic_id(doc: Document, fallback_idx: int) -> str:
    source = doc.metadata.get("source_file", "unknown")
    chunk_type = doc.metadata.get("chunk_type", "text")
    idx = doc.metadata.get("chunk_index", doc.metadata.get("table_index", fallback_idx))
    key = f"{source}::{chunk_type}::{idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(QDRANT_COLLECTION):
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        sparse_vectors_config={
            "langchain-sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f"Created Qdrant hybrid collection: {QDRANT_COLLECTION}")


def get_vectorstore() -> QdrantVectorStore:
    client = get_qdrant_client()
    ensure_collection(client)

    dense_embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="langchain-sparse",
    )


def index_documents(docs: list[Document]) -> int:
    if not docs:
        return 0

    vectorstore = get_vectorstore()
    ids = [_make_deterministic_id(doc, i) for i, doc in enumerate(docs)]
    vectorstore.add_documents(documents=docs, ids=ids)
    return len(docs)


def load_chunks_from_json(json_path: Path) -> list[Document]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in raw
    ]


if __name__ == "__main__":
    chunks_file = CHUNKS_DATA_DIR / "tcs_fy26_financials_chunks.json"
    docs = load_chunks_from_json(chunks_file)
    print(f"Loaded {len(docs)} chunks from {chunks_file.name}")

    count = index_documents(docs)
    print(f"Indexed {count} chunks into Qdrant collection '{QDRANT_COLLECTION}'")

    # Quick sanity check query
    vs = get_vectorstore()
    results = vs.similarity_search("TCS FY26 net profit excluding exceptional items", k=3)
    print("\n--- Top 3 Hybrid Retrieval Results ---")
    for i, doc in enumerate(results, start=1):
        print(f"\n[{i}] type={doc.metadata.get('chunk_type')} | source={doc.metadata.get('source_file')}")
        print(doc.page_content[:250].replace("\n", " "))