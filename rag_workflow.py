# from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from langgraph.graph import END, StateGraph

from chains.evaluate import evaluate_docs
from chains.generate_answer import generate_chain

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

# llm = ChatOllama(model="llama3.2:1b")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
retriever = vectorstore.as_retriever()


def retrieve(state):
    question = state["question"]

    docs = retriever.invoke(question)
    return {"documents": docs, "question": question}


def evaluate(state):
    question = state["question"]
    documents = state["documents"]

    document_evaluations = []
    filtered_docs = []

    for doc in documents:
        response = evaluate_docs.invoke({"question": question, "document": doc.page_content})
        document_evaluations.append(response)

        result = response.score
        if result.lower() == "yes":
            filtered_docs.append(doc)

    return {
        "documents": filtered_docs,
        "question": question,
        "document_evaluations": document_evaluations,
    }


def generate_answer(state):
    question = state["question"]
    documents = state["documents"]

    solution = generate_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "solution": solution}


class GraphState(TypedDict):
    question: str
    solution: str
    documents: List[str]
    document_evaluations: Optional[List[Dict[str, Any]]]
    document_relevance_score: Optional[Dict[str, Any]]
    question_relevance_score: Optional[Dict[str, Any]]


def create_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("Retrieve Documents", retrieve)
    workflow.add_node("Grade Documents", evaluate)
    workflow.add_node("Generate Answer", generate_answer)

    workflow.set_entry_point("Retrieve Documents")
    workflow.add_edge("Retrieve Documents", "Grade Documents")
    workflow.add_edge("Grade Documents", "Generate Answer")
    # workflow.add_edge("Retrieve Documents", "Generate Answer")

    return workflow.compile()


graph = create_graph()


def process_question(question):
    result = graph.invoke(input={"question": question})
    return result
