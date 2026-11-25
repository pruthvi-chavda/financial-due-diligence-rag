from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="Financial question to answer")
    source_file: str | None = Field(
        default=None,
        description="Optional PDF filename filter (e.g., 'tcs_fy26_financials.pdf')",
    )
    top_n: int = Field(default=5, ge=1, le=10, description="Number of reranked chunks to pass to LLM")


class SourceCitation(BaseModel):
    source_file: str
    chunk_type: str
    index: int | str | None = None
    rerank_score: float
    snippet: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceCitation]
    model_used: str
    guardrail_triggered: bool = False


class IngestResponse(BaseModel):
    source_file: str
    chunks_indexed: int
    status: str


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    collection_name: str
    indexed_points: int
