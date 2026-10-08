from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from practicals.llm_utils import get_embeddings

def calculate_similarity(client: OpenAI, text1: str, text2: str, model: str):
    vectors = get_embeddings(client, [text1, text2], model)
    score = float(cosine_similarity([vectors[0]], [vectors[1]])[0][0])
    return {
        "text1": text1,
        "text2": text2,
        "cosine_similarity": round(score, 6)
    }
