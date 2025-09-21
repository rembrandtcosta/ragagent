import streamlit as st
from langchain_ollama import ChatOllama
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from rag_workflow import process_question


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

llm = ChatOllama(model="llama3.2:1b")
retriever = vectorstore.as_retriever()

system_prompt = (
    "Você é um especialista em legislação condominial brasileira."
    "Sua tarefa é responder às perguntas dos clientes da forma mais verdadeira possível."
    "Use as informações recuperadas da legislação condominial brasileira para ajudar a responder às perguntas."
    "Seu desempenho é crítico para o bem-estar dos clientes e o sucesso de sua carreira."
    "Se você não souber a resposta, diga que não sabe."
    "Use no máximo três frases e mantenha a resposta concisa."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt), ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

st.title("Assistente de Legislação Condominial Brasileira")

query = st.chat_input(placeholder="O que você gostaria de saber?")
if query:
    with st.chat_message("user"):
        st.write(query)

    result = process_question(query)
    print(result)
    with st.chat_message("assistant"):
        st.write(result["solution"])
        st.write("**Fonte:**")
        for doc in result["documents"]:
            st.write(f'- {doc.page_content}')
