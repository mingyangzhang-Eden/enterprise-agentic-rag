import logging
import uuid

from fastapi import FastAPI, HTTPException

from api.schemas import QueryRequest, QueryResponse
from api.service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s " "%(levelname)s " "%(name)s " "%(message)s"),
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Enterprise Agentic RAG API",
    description=("REST API for the V3.2 " "Source-Aware Agentic RAG system."),
    version="1.0.0",
)


rag_service: RAGService | None = None


@app.on_event("startup")
def startup_event():
    global rag_service

    logger.info("application_starting")

    try:
        rag_service = RAGService()

    except Exception:
        logger.exception("rag_service_initialization_failed")

        raise

    logger.info("application_ready")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "enterprise-agentic-rag",
        "version": "v3.2",
        "rag_loaded": rag_service is not None,
    }


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query_rag(
    request: QueryRequest,
):
    request_id = str(uuid.uuid4())

    logger.info(
        "request_started request_id=%s endpoint=/query",
        request_id,
    )

    if rag_service is None:
        logger.warning(
            "request_rejected request_id=%s reason=service_not_ready",
            request_id,
        )

        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready.",
        )

    try:
        result = rag_service.query(
            question=request.question,
            request_id=request_id,
        )

    except Exception:
        logger.exception(
            "request_failed request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The RAG service could not complete "
                "the request. Please try again later."
            ),
        )

    logger.info(
        "request_completed request_id=%s status=200",
        request_id,
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        decision_steps=result["decision_steps"],
        tool_actions=result["tool_actions"],
        latency_seconds=result["latency_seconds"],
        request_id=request_id,
        evidence_sufficient=result["evidence_sufficient"],
    )
