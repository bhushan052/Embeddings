from openai import OpenAI
from practicals.llm_utils import get_embeddings

def generate_embeddings(client: OpenAI, texts: list[str], model: str):
    return get_embeddings(client, texts, model)
