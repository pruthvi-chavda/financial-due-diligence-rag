from langchain_core.documents import Document
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sentence_transformers import CrossEncoder

from src.vectorstore import get_vectorstore

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Lazy-load and cache the Cross-Encoder on CPU to avoid GPU heat and reload latency."""
    global _reranker
    if _reranker is None:
        print(f"Loading Cross-Encoder reranker ({RERANKER_MODEL_NAME}) on CPU...")
        _reranker = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")
    return _reranker


def _build_source_filter(source_file: str | None) -> Filter | None:
    """Build a Qdrant payload filter scoped to a specific ingested PDF."""
    if not source_file:
        return None

    return Filter(
        must=[
            FieldCondition(
                key="metadata.source_file",
                match=MatchValue(value=source_file),
            )
        ]
    )


def retrieve_documents(
    query: str,
    source_file: str | None = None,
    initial_k: int = 15,
    top_n: int = 5,
) -> list[Document]:
    """
    Two-stage financial retrieval:
      1. Broad hybrid search (Dense + BM25) in Qdrant with optional source_file filter.
      2. Cross-Encoder reranking on CPU to surface exact financial tables and figures.
    """
    vectorstore = get_vectorstore()
    query_filter = _build_source_filter(source_file)

   
    candidates = vectorstore.similarity_search(
        query=query,
        k=initial_k,
        filter=query_filter,
    )
    if not candidates:
        return []


    reranker = _get_reranker()
    pairs = [(query, doc.page_content) for doc in candidates]
    raw_scores = reranker.predict(pairs)

    scored_docs: list[Document] = []
    for doc, score in zip(candidates, raw_scores):
        
        doc.metadata["rerank_score"] = round(float(score), 4)
        scored_docs.append(doc)

    
    scored_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    return scored_docs[:top_n]


if __name__ == "__main__":
    
    test_query = "What were TCS consolidated exceptional items and net profit in FY26?"
    print(f"Query: {test_query}\n")

    results = retrieve_documents(
        query=test_query,
        source_file="tcs_fy26_financials.pdf",
        initial_k=15,
        top_n=5,
    )

    print(f"--- Top {len(results)} Reranked Results ---")
    for rank, doc in enumerate(results, start=1):
        score = doc.metadata.get("rerank_score")
        ctype = doc.metadata.get("chunk_type")
        idx = doc.metadata.get("table_index", doc.metadata.get("chunk_index"))
        preview = doc.page_content[:220].replace("\n", " ")
        print(f"\n[{rank}] score={score} | type={ctype} | index={idx}")
        print(f"    {preview}...")

  
    isolated = retrieve_documents(
        query="Revenue from operations",
        source_file="non_existent_company.pdf",
    )
    print(f"\nIsolation check (should be 0): {len(isolated)} results")