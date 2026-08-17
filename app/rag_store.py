import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "sales_knowledge"

_model = SentenceTransformer(MODEL_NAME)
_chroma_client = chromadb.Client()
_collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def load_paragraphs():
    knowledge_path = Path("data/sales_knowledge.txt")
    knowledge = knowledge_path.read_text(encoding="utf-8")
    return [paragraph.strip() for paragraph in knowledge.split("\n\n") if paragraph.strip()]


def find_probability_context(question: str, paragraphs: list[str]):
    percentage_match = re.search(r"\b(\d{1,3})\s*percent\b", question.lower())
    if not percentage_match:
        return ""

    probability = int(percentage_match.group(1))

    if probability >= 70:
        return next(
            paragraph for paragraph in paragraphs
            if "High probability" in paragraph
        )

    if probability >= 40:
        return next(
            paragraph for paragraph in paragraphs
            if "Medium probability" in paragraph
        )

    return next(
        paragraph for paragraph in paragraphs
        if "Low probability" in paragraph
    )


def build_vector_store():
    paragraphs = load_paragraphs()

    if _collection.count() > 0:
        return

    embeddings = _model.encode(paragraphs).tolist()
    ids = [f"sales-doc-{index}" for index in range(len(paragraphs))]

    _collection.add(
        ids=ids,
        documents=paragraphs,
        embeddings=embeddings,
    )


def search_sales_knowledge(question: str):
    paragraphs = load_paragraphs()

    probability_context = find_probability_context(question, paragraphs)
    if probability_context:
        return probability_context

    build_vector_store()

    question_embedding = _model.encode([question]).tolist()[0]

    results = _collection.query(
        query_embeddings=[question_embedding],
        n_results=1,
    )

    documents = results.get("documents", [[]])
    if not documents or not documents[0]:
        return ""

    return documents[0][0]