from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)


class EvaluateDocs(BaseModel):
    score: str = Field(
        description="Se os documentos são relevantes para a pergunta - 'sim' se relevantes, 'não' se não relevantes",
    )


structured_output = llm.with_structured_output(EvaluateDocs)

system = """Você é um avaliador especialista em relevância de documentos para um sistema RAG (Retrieval-Augmented Generation). Seu papel é avaliar se os documentos recuperados contêm informações suficientes para responder de forma eficaz à consulta do usuário.

QUADRO DE AVALIAÇÃO:

1. RELEVÂNCIA TEMÁTICA:

Os documentos abordam diretamente o tema principal da consulta?
Os conceitos e tópicos-chave estão alinhados com o que o usuário está perguntando?

2. QUALIDADE DA INFORMAÇÃO:

As informações são precisas e confiáveis?
Há declarações conflitantes nos documentos?
As informações estão atualizadas e são relevantes para o contexto da consulta?


CRITÉRIOS DE PONTUAÇÃO:

Marque “sim” se os documentos fornecerem informações relevantes para responder satisfatoriamente à consulta.

Marque “não” se os documentos carecerem de informações essenciais ou estiverem fora de tópico. 

REQUISITOS ADICIONAIS:

Forneça uma pontuação de relevância (0,0–1,0) indicando a qualidade da correspondência.
Avalie a cobertura dos requisitos da consulta.
Identifique qualquer informação crítica ausente.
Seja minucioso, mas eficiente em sua avaliação. Foque na utilidade prática para a geração de respostas."""

human_prompt = """Por favor, avalie se os documentos recuperados são suficientes para responder à consulta do usuário.

USER QUERY:
{question}

RETRIEVED DOCUMENTS:
{document}

AVALIAÇÃO REQUERIDA:
1. Pontuação Primária: 'sim' se os documentos forem relevantes, 'não' se não forem relevantes
2. Pontuação de Relevância: avaliação de 0,0–1,0 de quão bem os documentos correspondem à consulta
3. Avaliação de Cobertura: Quão bem os documentos atendem aos requisitos da consulta?
4. Informação Ausente: Quais informações-chave (se houver) estão faltando para uma resposta completa?

Forneça sua avaliação abrangente com base no quadro acima.
"""

evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", human_prompt),
    ]
)

evaluate_docs = evaluate_prompt | structured_output
