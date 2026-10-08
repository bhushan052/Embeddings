import logging
import os
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from practicals.generate_embeddings import generate_embeddings
from practicals.semantic_similarity import calculate_similarity
from practicals.semantic_search import semantic_search

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not configured in .env")

client = OpenAI(api_key=API_KEY)

logger = logging.getLogger("module6")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)

app = FastAPI(title="Module 6 - Embeddings & Semantic Search API", version="1.0.0")

class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)

class SimilarityRequest(BaseModel):
    text1: str
    text2: str

class SemanticSearchRequest(BaseModel):
    query: str
    documents: list[str] = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1)

@app.get("/")
def root():
    return {
        "message": "Module 6 Embeddings API is running",
        "docs": "/docs",
        "embedding_model": EMBEDDING_MODEL
    }

@app.post("/embeddings")
def embeddings(request: EmbeddingRequest):
    try:
        vectors = generate_embeddings(client, request.texts, EMBEDDING_MODEL)
        return {
            "model": EMBEDDING_MODEL,
            "count": len(vectors),
            "dimension": len(vectors[0]) if vectors else 0,
            "embeddings": vectors
        }
    except Exception as exc:
        logger.exception("Embedding generation failed")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/similarity")
def similarity(request: SimilarityRequest):
    try:
        return calculate_similarity(client, request.text1, request.text2, EMBEDDING_MODEL)
    except Exception as exc:
        logger.exception("Similarity calculation failed")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/semantic-search")
def search(request: SemanticSearchRequest):
    try:
        return {
            "query": request.query,
            "results": semantic_search(
                client, request.query, request.documents,
                request.top_k, EMBEDDING_MODEL
            )
        }
    except Exception as exc:
        logger.exception("Semantic search failed")
        raise HTTPException(status_code=500, detail=str(exc))
