import os
import time
import psutil
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from config import CHUNK_SIZE, CHUNK_OVERLAP, CLAUSE_CHUNK_SIZE, CLAUSE_CHUNK_OVERLAP, MODEL_NAME
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from langgraph.graph import StateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import CrossEncoder
from chains.generate_answer import generate_chain
from chains.internal_docs import internal_docs
from chains.clause_extractor import clause_extractor_chain, deduplicate_clauses, ExtractedClause
from chains.illegality_detector import illegality_detector_chain

from typing import List, Dict, Any, Optional, TypedDict

import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="retrieval_document"
)

vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./db",
        client_settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True,
        )
)

retriever_internal = None


def set_internal_retriever(document):
    global retriever_internal
    with open("internal_doc.pdf", "wb") as f:
        f.write(document)
    loader = PyPDFLoader("internal_doc.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = text_splitter.split_documents(docs)
    internal_vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./db_internal",
        client_settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True,
        )
    )

    retriever_internal = internal_vectorstore.as_retriever()


llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
retriever = vectorstore.as_retriever()


def retrieve(state):
    question = state["question"]

    docs = retriever.invoke(question)
    return {"documents": docs, "question": question}


def retrieve_internal_docs(state):
    question = state["question"]

    docs = retriever_internal.invoke(question)
    return {"internal_documents": docs}


reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def evaluate(state):
    question = state["question"]
    documents = state["documents"]

    doc_pairs = [[question, doc.page_content] for doc in documents]

    scores = reranker_model.predict(doc_pairs)

    filtered_docs = []

    THRESHOLD = -3.0
    for doc, score in zip(documents, scores):
        if score > THRESHOLD:
            filtered_docs.append(doc)

    return {"documents": filtered_docs, "question": question}


def internal(state):
    question = state["question"]
    internal_documents = state["internal_documents"]
    documents = state["documents"]

    for doc in internal_documents:
        internal_relevance = internal_docs.invoke({
            "question": question,
            "document": doc
        })
        if internal_relevance.score.lower() == "sim":
            documents = documents + [doc]

    return {
        "documents": documents,
        "question": question,
        "document_evaluations": state.get("document_evaluations", []),
    }


def generate_answer(state):
    question = state["question"]
    documents = state["documents"]

    solution = generate_chain.invoke({
        "context": documents,
        "question": question,
    })
    return {"documents": documents, "question": question, "solution": solution}


class GraphState(TypedDict):
    question: str
    solution: str
    documents: List[str]
    internal_documents: Optional[List[str]]
    document_evaluations: Optional[List[Dict[str, Any]]]
    document_relevance_score: Optional[Dict[str, Any]]
    question_relevance_score: Optional[Dict[str, Any]]


def create_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("Retrieve Documents", retrieve)
    workflow.add_node("Retrieve Internal Documents", retrieve_internal_docs)
    workflow.add_node("Grade Documents", evaluate)
    workflow.add_node("Check Internal Docs", internal)
    workflow.add_node("Generate Answer", generate_answer)

    workflow.set_entry_point("Retrieve Documents")
    workflow.add_edge("Retrieve Documents", "Generate Answer")
    workflow.add_conditional_edges(
        "Grade Documents",
        lambda state:
            "answer" if retriever_internal is None else "internal_docs",
        {
            "answer": "Generate Answer",
            "internal_docs": "Retrieve Internal Documents"
        }
    )
    workflow.add_edge("Retrieve Internal Documents", "Check Internal Docs")
    workflow.add_edge("Check Internal Docs", "Generate Answer")

    return workflow.compile()


graph = create_graph()


def generate_graph_diagram():
    graph.get_graph().draw_png("rag_workflow_diagram.png")


def _get_footprint() -> Dict[str, float]:
    process = psutil.Process(os.getpid())
    return {
        "memory": process.memory_info().rss / (1024 * 1024),  # in MB
        "cpu": process.cpu_times(),
    }


def _get_diff_footprint(before: Dict[str, float], after: Dict[str, float]
                        ) -> Dict[str, float]:
    return {
        "memory_diff": after["memory"] - before["memory"],
        "cpu_diff": after["cpu"].user - before["cpu"].user,
    }


@contextmanager
def timer(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label} Elapsed time: {end - start:.4f} seconds")


def process_question(question):
    before = _get_footprint()
    with timer("RAG Workflow"):
        result = graph.invoke(input={"question": question})
    after = _get_footprint()
    footprint = _get_diff_footprint(before, after)
    return result, footprint


# Data models for document analysis
@dataclass
class ClauseAnalysisResult:
    clause_number: str
    clause_text: str
    topic: str
    is_potentially_illegal: bool
    confidence: str  # alta, media, baixa
    conflicting_articles: List[str]
    explanation: str
    legal_principle_violated: Optional[str]
    recommendation: str


@dataclass
class DocumentAnalysisReport:
    document_name: str
    analysis_date: str
    total_clauses_analyzed: int
    potentially_illegal_count: int
    clauses: List[ClauseAnalysisResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_name": self.document_name,
            "analysis_date": self.analysis_date,
            "total_clauses_analyzed": self.total_clauses_analyzed,
            "potentially_illegal_count": self.potentially_illegal_count,
            "clauses": [asdict(c) for c in self.clauses]
        }


# Analysis workflow state
class AnalysisState(TypedDict):
    document_chunks: List[str]
    extracted_clauses: List[Dict[str, Any]]
    current_clause_index: int
    relevant_articles: str
    analysis_results: List[Dict[str, Any]]


def extract_clauses_node(state: AnalysisState) -> AnalysisState:
    """Extract clauses from all document chunks."""
    all_clauses = []
    for chunk in state["document_chunks"]:
        try:
            result = clause_extractor_chain.invoke({"document_chunk": chunk})
            for clause in result.clauses:
                all_clauses.append({
                    "clause_number": clause.clause_number,
                    "clause_text": clause.clause_text,
                    "topic": clause.topic
                })
        except Exception as e:
            print(f"Error extracting clauses from chunk: {e}")
            continue

    # Deduplicate based on clause_number
    seen = set()
    unique_clauses = []
    for clause in all_clauses:
        if clause["clause_number"] not in seen:
            seen.add(clause["clause_number"])
            unique_clauses.append(clause)

    return {
        **state,
        "extracted_clauses": unique_clauses,
        "current_clause_index": 0,
        "analysis_results": []
    }


def retrieve_relevant_articles_node(state: AnalysisState) -> AnalysisState:
    """Retrieve relevant Civil Code articles for the current clause."""
    if state["current_clause_index"] >= len(state["extracted_clauses"]):
        return state

    clause = state["extracted_clauses"][state["current_clause_index"]]

    # Build query based on clause topic and content
    query = f"Código Civil {clause['topic']} condomínio {clause['clause_text'][:200]}"

    try:
        docs = retriever.invoke(query)
        relevant_articles = "\n\n".join([doc.page_content for doc in docs[:5]])
    except Exception as e:
        print(f"Error retrieving articles: {e}")
        relevant_articles = ""

    return {**state, "relevant_articles": relevant_articles}


def analyze_clause_node(state: AnalysisState) -> AnalysisState:
    """Analyze the current clause for potential illegality."""
    if state["current_clause_index"] >= len(state["extracted_clauses"]):
        return state

    clause = state["extracted_clauses"][state["current_clause_index"]]

    try:
        analysis = illegality_detector_chain.invoke({
            "clause_number": clause["clause_number"],
            "clause_topic": clause["topic"],
            "clause_text": clause["clause_text"],
            "relevant_articles": state["relevant_articles"]
        })

        result = {
            "clause_number": clause["clause_number"],
            "clause_text": clause["clause_text"],
            "topic": clause["topic"],
            "is_potentially_illegal": analysis.is_potentially_illegal,
            "confidence": analysis.confidence,
            "conflicting_articles": analysis.conflicting_articles,
            "explanation": analysis.explanation,
            "legal_principle_violated": analysis.legal_principle_violated,
            "recommendation": analysis.recommendation
        }
    except Exception as e:
        print(f"Error analyzing clause: {e}")
        result = {
            "clause_number": clause["clause_number"],
            "clause_text": clause["clause_text"],
            "topic": clause["topic"],
            "is_potentially_illegal": False,
            "confidence": "baixa",
            "conflicting_articles": [],
            "explanation": f"Erro na análise: {str(e)}",
            "legal_principle_violated": None,
            "recommendation": "Revisar manualmente"
        }

    new_results = state["analysis_results"] + [result]
    new_index = state["current_clause_index"] + 1

    return {**state, "analysis_results": new_results, "current_clause_index": new_index}


def should_continue(state: AnalysisState) -> str:
    """Determine if there are more clauses to analyze."""
    if state["current_clause_index"] < len(state["extracted_clauses"]):
        return "continue"
    return "complete"


def create_analysis_graph():
    """Create the document analysis workflow graph."""
    workflow = StateGraph(AnalysisState)

    workflow.add_node("Extract Clauses", extract_clauses_node)
    workflow.add_node("Retrieve Articles", retrieve_relevant_articles_node)
    workflow.add_node("Analyze Clause", analyze_clause_node)

    workflow.set_entry_point("Extract Clauses")
    workflow.add_edge("Extract Clauses", "Retrieve Articles")
    workflow.add_edge("Retrieve Articles", "Analyze Clause")

    workflow.add_conditional_edges(
        "Analyze Clause",
        should_continue,
        {
            "continue": "Retrieve Articles",
            "complete": "__end__"
        }
    )

    return workflow.compile()


analysis_graph = create_analysis_graph()


def analyze_document(document_bytes: bytes, document_name: str) -> DocumentAnalysisReport:
    """Analyze a condominium document for potentially illegal clauses."""
    # Save document temporarily
    temp_path = "temp_analysis_doc.pdf"
    with open(temp_path, "wb") as f:
        f.write(document_bytes)

    try:
        # Load and split document
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CLAUSE_CHUNK_SIZE,
            chunk_overlap=CLAUSE_CHUNK_OVERLAP,
        )
        splits = text_splitter.split_documents(docs)
        chunks = [doc.page_content for doc in splits]

        # Run analysis workflow
        initial_state: AnalysisState = {
            "document_chunks": chunks,
            "extracted_clauses": [],
            "current_clause_index": 0,
            "relevant_articles": "",
            "analysis_results": []
        }

        # Set high recursion limit to handle documents with many clauses
        # Each clause requires 2 iterations (retrieve + analyze)
        result = analysis_graph.invoke(
            initial_state,
            config={"recursion_limit": 500}
        )

        # Build report
        clause_results = [
            ClauseAnalysisResult(
                clause_number=r["clause_number"],
                clause_text=r["clause_text"],
                topic=r["topic"],
                is_potentially_illegal=r["is_potentially_illegal"],
                confidence=r["confidence"],
                conflicting_articles=r["conflicting_articles"],
                explanation=r["explanation"],
                legal_principle_violated=r["legal_principle_violated"],
                recommendation=r["recommendation"]
            )
            for r in result["analysis_results"]
        ]

        illegal_count = sum(1 for c in clause_results if c.is_potentially_illegal)

        report = DocumentAnalysisReport(
            document_name=document_name,
            analysis_date=datetime.now().isoformat(),
            total_clauses_analyzed=len(clause_results),
            potentially_illegal_count=illegal_count,
            clauses=clause_results
        )

        return report

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
