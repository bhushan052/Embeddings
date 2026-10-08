from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from practicals.llm_utils import get_embeddings

def semantic_search(client: OpenAI, query: str, documents: list[str], top_k: int, model: str):
    vectors = get_embeddings(client, [query] + documents, model)
    scores = cosine_similarity([vectors[0]], vectors[1:])[0]

    ranked = sorted(
        [
            {"document": doc, "score": round(float(score), 6)}
            for doc, score in zip(documents, scores)
        ],
        key=lambda x: x["score"],
        reverse=True
    )
    return ranked[:min(top_k, len(ranked))]
