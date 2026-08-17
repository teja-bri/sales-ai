import logging
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from langfuse import get_client
from pydantic import BaseModel, Field

from app.rag_store import search_sales_knowledge

load_dotenv()

if os.getenv("LANGFUSE_HOST") and not os.getenv("LANGFUSE_BASE_URL"):
    os.environ["LANGFUSE_BASE_URL"] = os.getenv("LANGFUSE_HOST")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

langfuse = get_client()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

app = FastAPI(
    title="Sales AI",
    version="0.1.0",
)


class OpportunityRequest(BaseModel):
    account_name: str
    opportunity_name: str
    stage: str
    value: float = Field(gt=0)
    probability: int = Field(ge=0, le=100)


class RagQuestion(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Sales AI CI/CD is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze-opportunity")
def analyze_opportunity(opportunity: OpportunityRequest):
    logger.info("Analyzing opportunity: %s", opportunity.opportunity_name)

    if opportunity.probability >= 70:
        risk_level = "low"
        recommended_action = "Prepare final proposal and confirm closing timeline."
    elif opportunity.probability >= 40:
        risk_level = "medium"
        recommended_action = "Schedule a decision-maker follow-up within 3 days."
    else:
        risk_level = "high"
        recommended_action = "Review blockers, budget fit, and customer urgency."

    summary = (
        f"{opportunity.opportunity_name} for {opportunity.account_name} "
        f"is in {opportunity.stage} stage with {opportunity.probability}% probability."
    )

    return {
        "risk_level": risk_level,
        "summary": summary,
        "recommended_action": recommended_action,
    }


def load_sales_knowledge():
    knowledge_path = Path("data/sales_knowledge.txt")
    return knowledge_path.read_text(encoding="utf-8")


def find_relevant_context(question: str, knowledge: str):
    paragraphs = [paragraph.strip() for paragraph in knowledge.split("\n\n") if paragraph.strip()]

    percentage_match = re.search(r"\b(\d{1,3})\s*percent\b", question.lower())
    if percentage_match:
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

    question_words = set(question.lower().split())

    best_paragraph = ""
    best_score = 0

    for paragraph in paragraphs:
        paragraph_words = set(paragraph.lower().split())
        score = len(question_words.intersection(paragraph_words))

        if score > best_score:
            best_score = score
            best_paragraph = paragraph

    return best_paragraph


def build_rag_prompt(question: str, context: str):
    return f"""
You are a sales AI assistant. Answer only using the provided context.

Context:
{context}

Question:
{question}

Answer in 3 short bullet points. If the context is not enough, say what is missing.
"""


@app.post("/rag-answer")
def rag_answer(request: RagQuestion):
    logger.info("Answering RAG question with vector search: %s", request.question)

    trace_id = langfuse.create_trace_id()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-answer",
        trace_context={"trace_id": trace_id},
        input={"question": request.question},
    ) as span:
        context = search_sales_knowledge(request.question)

        if not context:
            logger.warning("No relevant context found for question: %s", request.question)

            output = {
                "answer": "I could not find relevant sales knowledge for this question.",
                "context": "",
            }

            span.update(output=output)

            span.score(
                name="retrieval_found",
                value=0,
                comment="RAG did not retrieve a context paragraph",
            )

            langfuse.flush()

            return output

        logger.info("Retrieved vector context for RAG question")

        output = {
            "answer": f"Based on the sales knowledge: {context}",
            "context": context,
        }

        span.update(
            output=output,
            metadata={"retrieved_context": context},
        )

        span.score(
            name="retrieval_found",
            value=1,
            comment="RAG retrieved a context paragraph",
        )

        langfuse.flush()

        return output


@app.post("/rag-generate")
async def rag_generate(request: RagQuestion):
    logger.info("Generating RAG answer with Qwen: %s", request.question)

    trace_id = langfuse.create_trace_id()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-generate",
        trace_context={"trace_id": trace_id},
        input={"question": request.question},
    ) as span:
        context = search_sales_knowledge(request.question)
        prompt = build_rag_prompt(request.question, context)

        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_predict": 250,
                    },
                },
            )

        response.raise_for_status()
        llm_output = response.json()["response"]

        output = {
            "answer": llm_output,
            "context": context,
            "model": OLLAMA_MODEL,
        }

        span.update(
            output=output,
            metadata={
                "retrieved_context": context,
                "model": OLLAMA_MODEL,
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 250,
            },
        )

        langfuse.flush()

        return output