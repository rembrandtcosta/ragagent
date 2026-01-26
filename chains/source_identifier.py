from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import MODEL_NAME

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


class UsedSources(BaseModel):
    used_indices: List[int] = Field(
        description="Lista dos índices (começando em 0) dos documentos que foram efetivamente usados na resposta"
    )


structured_output = llm.with_structured_output(UsedSources)

system_prompt = """Você é um analisador de citações de fontes.

Sua tarefa é identificar quais documentos de uma lista foram DIRETAMENTE usados para responder a PERGUNTA ESPECÍFICA do usuário.

Um documento deve ser incluído APENAS se:
- Ele contém a informação PRINCIPAL que responde diretamente à pergunta
- A resposta cita ou parafraseia informação essencial desse documento para responder à pergunta

Um documento NÃO deve ser incluído se:
- Ele contém informações complementares ou tangenciais que não eram necessárias para responder à pergunta
- A resposta menciona o documento apenas como contexto adicional, não como fonte principal
- O documento trata de assuntos relacionados mas não diretamente perguntados

IMPORTANTE: Seja MUITO RESTRITIVO. Inclua apenas os documentos que contêm a resposta DIRETA à pergunta.
Se a pergunta é sobre "duração do mandato", inclua apenas documentos que falam sobre duração do mandato.
Se nenhum documento foi usado diretamente, retorne uma lista vazia."""

human_prompt = """Analise quais documentos foram usados para responder DIRETAMENTE à pergunta do usuário.

PERGUNTA DO USUÁRIO:
{question}

RESPOSTA GERADA:
{answer}

DOCUMENTOS DISPONÍVEIS:
{documents}

Retorne APENAS os índices dos documentos que contêm informação DIRETAMENTE relevante para responder à pergunta específica do usuário."""

identify_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
])

source_identifier_chain = identify_prompt | structured_output


def identify_used_sources(answer: str, documents: list, question: str = "") -> list:
    """
    Identify which documents were actually used to answer the question.

    Args:
        answer: The generated answer text
        documents: List of document objects
        question: The original user question

    Returns:
        List of documents that were directly used to answer the question
    """
    if not documents or not answer:
        return []

    # Format documents with indices
    docs_text = "\n\n".join([
        f"[Documento {i}]: {doc.page_content}"
        for i, doc in enumerate(documents)
    ])

    try:
        result = source_identifier_chain.invoke({
            "question": question,
            "answer": answer,
            "documents": docs_text
        })

        # Return only the used documents
        used_docs = [
            documents[i] for i in result.used_indices
            if 0 <= i < len(documents)
        ]
        return used_docs

    except Exception as e:
        print(f"Error identifying sources: {e}")
        # Fallback: return all documents
        return documents
