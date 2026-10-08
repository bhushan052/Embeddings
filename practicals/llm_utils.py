from openai import OpenAI

def get_embeddings(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    if not texts:
        return []
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]
