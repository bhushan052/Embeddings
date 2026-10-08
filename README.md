# Module 6 — Embeddings, Semantic Similarity & Semantic Search API

## Structure

```text
module6_embeddings_api/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── app.py
├── app.log
└── practicals/
    ├── __init__.py
    ├── llm_utils.py
    ├── generate_embeddings.py
    ├── semantic_similarity.py
    └── semantic_search.py
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`:

```env
OPENAI_API_KEY=your_real_api_key
EMBEDDING_MODEL=text-embedding-3-small
```

Run:

```bash
uvicorn app:app --reload
```

Swagger: `http://127.0.0.1:8000/docs`

## Postman

### 1. Generate Embeddings

POST `http://127.0.0.1:8000/embeddings`

```json
{
  "texts": [
    "I like programming.",
    "I enjoy coding.",
    "I went to the beach."
  ]
}
```

### 2. Semantic Similarity

POST `http://127.0.0.1:8000/similarity`

```json
{
  "text1": "I like programming.",
  "text2": "I enjoy coding."
}
```

Try unrelated:

```json
{
  "text1": "I like programming.",
  "text2": "I went to the beach."
}
```

### 3. Semantic Search

POST `http://127.0.0.1:8000/semantic-search`

```json
{
  "query": "How can I recover my account?",
  "documents": [
    "Follow these steps to reset your forgotten password.",
    "You can update your email address from account settings.",
    "Two-factor authentication protects your account.",
    "The company picnic will take place next Friday."
  ],
  "top_k": 3
}
```

Pipeline:

```text
Postman
  ↓
FastAPI
  ↓
Practical Module
  ↓
OpenAI Embedding API
  ↓
Text → Vector
  ↓
Cosine Similarity
  ↓
Ranking
  ↓
JSON Response
```
