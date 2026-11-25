import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.config import PROCESSED_DATA_DIR, QDRANT_COLLECTION
from src.ingestion import ingest_pdf  
from src.rag_chain import run_rag_pipeline
from src.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from src.vectorstore import get_qdrant_client, index_documents

RAW_DATA_DIR = PROCESSED_DATA_DIR.parent / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Financial Due Diligence RAG API",
    description="Hybrid Qdrant + Cross-Encoder Reranked Financial Filing QA System",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Verify Qdrant connectivity and report total indexed vector points."""
    try:
        client = get_qdrant_client()
        if client.collection_exists(QDRANT_COLLECTION):
            point_count = client.count(collection_name=QDRANT_COLLECTION).count
        else:
            point_count = 0

        return HealthResponse(
            status="ok",
            qdrant_connected=True,
            collection_name=QDRANT_COLLECTION,
            indexed_points=point_count,
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            qdrant_connected=False,
            collection_name=QDRANT_COLLECTION,
            indexed_points=0,
        )


@app.get("/documents", response_model=list[str])
def list_indexed_documents() -> list[str]:
    """Return sorted list of unique PDF source_file names stored in Qdrant."""
    try:
        client = get_qdrant_client()
        if not client.collection_exists(QDRANT_COLLECTION):
            return []

        records, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1000,
            with_payload=["metadata.source_file"],
            with_vectors=False,
        )
        files = {
            rec.payload.get("metadata", {}).get("source_file")
            for rec in records
            if rec.payload
        }
        files.discard(None)
        return sorted(files)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {exc}")


@app.post("/query", response_model=QueryResponse)
def query_filings(request: QueryRequest) -> QueryResponse:
    """Run guarded hybrid retrieval, reranking, and grounded LLM synthesis."""
    try:
        return run_rag_pipeline(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}")


@app.post("/ingest", response_model=IngestResponse)
def ingest_filing(file: UploadFile = File(...)) -> IngestResponse:
    """Upload a PDF filing, parse/chunk via Docling, and index into Qdrant."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    safe_filename = Path(file.filename).name
    dest_path = RAW_DATA_DIR / safe_filename

    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer) 

        docs = ingest_pdf(dest_path)
        if not docs:
            return IngestResponse(
                source_file=safe_filename,
                chunks_indexed=0,
                status="skipped",
            )

        count = index_documents(docs)
        return IngestResponse(
            source_file=safe_filename,
            chunks_indexed=count,
            status="success",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        file.file.close()