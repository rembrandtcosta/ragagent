from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_ollama import ChatOllama
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode

llm = ChatOllama(model="llama3.2:1b")

embeddings = NomicEmbeddings(
        model="nomic-embed-text-v1.5",
        inference_mode="local",
)

vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./db",
        client_settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True,
        )
)

retriever = vectorstore.as_retriever()
retriever_tool = create_retriever_tool(retriever, "buscador jurídico", "Use este ferramenta para buscar informações na legislação condominial brasileira.")
retriever_tool.invoke({"query": "Quem paga por reformas no condomínio?"})


def interpret_query(state: MessagesState):
    response = (
        llm
        .bind_tools([retriever_tool])
        .invoke(state["messages"])
    )
    return {"messages": [response]}


def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = (
        "Você é um especialista em legislação condominial brasileira."
        "Sua tarefa é responder às perguntas dos clientes da forma mais verdadeira possível."
        "Use as informações recuperadas da legislação condominial brasileira para ajudar a responder às perguntas."
        "Seu desempenho é crítico para o bem-estar dos clientes e o sucesso de sua carreira."
        "Se você não souber a resposta, diga que não sabe."
        "Use no máximo três frases e mantenha a resposta concisa."
        "\n\n"
        f"Contexto: {context}\n\n"
        f"Pergunta: {question}\n\n"
        "Resposta:"
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node(interpret_query)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(generate_answer)

workflow.add_edge(START, "interpret_query")
workflow.add_edge("interpret_query", "retrieve")
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)

graph = workflow.compile()


for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Quem paga por reformas no condomínio?",
            }
        ]
    }
):
    for node, update in chunk.items():
        print(update)
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")

