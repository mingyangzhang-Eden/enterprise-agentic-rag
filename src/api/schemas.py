from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Natural-language question sent to the RAG system.",
    )


class SourceItem(BaseModel):
    document_id: str
    chunk_index: int | None = None
    evidence_source: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]

    decision_steps: int
    tool_actions: int

    latency_seconds: float
    request_id: str
    evidence_sufficient: bool
