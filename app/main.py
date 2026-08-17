import logging
import re
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.rag_store import search_sales_knowledge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@app.post("/rag-answer")
def rag_answer(request: RagQuestion):
    logger.info("Answering RAG question with vector search: %s", request.question)

    context = search_sales_knowledge(request.question)

    if not context:
        logger.warning("No relevant context found for question: %s", request.question)

        return {
            "answer": "I could not find relevant sales knowledge for this question.",
            "context": "",
        }

    logger.info("Retrieved vector context for RAG question")

    return {
        "answer": f"Based on the sales knowledge: {context}",
        "context": context,
    }