import os
import time
import psutil
from contextlib import contextmanager
from config import CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from langgraph.graph import StateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chains.evaluate import evaluate_docs
from chains.generate_answer import generate_chain
from chains.internal_docs import internal_docs

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
    print("Internal docs:", docs)
    return {"internal_documents": docs}


def evaluate(state):
    question = state["question"]
    documents = state["documents"]

    document_evaluations = []
    filtered_docs = []

    for doc in documents:
        print(doc)
        response = evaluate_docs.invoke({
            "question": question,
            "document": doc.page_content
        })
        print(response)
        document_evaluations.append(response)

        result = response.score
        if result.lower() == "sim":
            filtered_docs.append(doc)

    return {
        "documents": filtered_docs,
        "question": question,
        "document_evaluations": document_evaluations,
    }


def internal(state):
    question = state["question"]
    internal_documents = state["internal_documents"]
    documents = state["documents"]

    for doc in internal_documents:
        print(doc)
        internal_relevance = internal_docs.invoke({
            "question": question,
            "document": doc
        })
        print("Internal doc relevance:", internal_relevance)
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
    workflow.add_edge("Retrieve Documents", "Grade Documents")
    workflow.add_edge("Grade Documents", "Generate Answer")
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
    print("Processing question:", question)
    with timer("RAG Workflow"):
        result = graph.invoke(input={"question": question})
    after = _get_footprint()
    footprint = _get_diff_footprint(before, after)
    print(footprint)
    return result, footprint
