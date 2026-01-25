from typing import Optional
from config import MODEL_NAME
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


class DocumentSuggestion(BaseModel):
    should_suggest: bool = Field(
        description="Se deve sugerir a redação de um documento"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Tipo de documento sugerido: notificacao_barulho, notificacao_inadimplencia, advertencia, convocacao_assembleia, ata_assembleia, comunicado_geral"
    )
    document_name: Optional[str] = Field(
        default=None,
        description="Nome amigável do documento em português (ex: 'Notificação de Barulho')"
    )
    suggestion_message: Optional[str] = Field(
        default=None,
        description="Mensagem sugerindo a redação do documento"
    )


structured_output = llm.with_structured_output(DocumentSuggestion)

system_prompt = """Você é um assistente especializado em documentos condominiais brasileiros.

Sua tarefa é analisar a pergunta do usuário e determinar se é apropriado sugerir a redação de um documento formal.

TIPOS DE DOCUMENTOS QUE VOCÊ PODE SUGERIR:

1. notificacao_barulho - Notificação de Barulho
   - Quando: perguntas sobre barulho excessivo, perturbação do sossego, vizinho barulhento, música alta, festas

2. notificacao_inadimplencia - Notificação de Inadimplência
   - Quando: perguntas sobre condômino devedor, taxa atrasada, cobrança de condomínio

3. advertencia - Advertência Formal
   - Quando: perguntas sobre infrações ao regimento, descumprimento de regras, comportamento inadequado

4. convocacao_assembleia - Convocação de Assembleia
   - Quando: perguntas sobre como convocar assembleia, reunião de condôminos, votação

5. ata_assembleia - Ata de Assembleia
   - Quando: perguntas sobre registro de assembleia, documentar reunião, ata

6. comunicado_geral - Comunicado Geral
   - Quando: perguntas sobre avisar moradores, comunicação geral, informar condôminos

DIRETRIZES:

1. Só sugira documento se a pergunta claramente se relaciona a uma situação prática
2. Perguntas puramente teóricas não devem gerar sugestões
3. A mensagem de sugestão deve ser natural e educada
4. Não sugira documento se o usuário já está pedindo explicitamente para redigir um

EXEMPLOS:

Pergunta: "O que fazer quando o vizinho faz muito barulho?"
-> Sugerir: notificacao_barulho
-> Mensagem: "Gostaria que eu redigisse uma Notificação de Barulho para enviar ao condômino?"

Pergunta: "Quais são as regras sobre animais no condomínio?"
-> Não sugerir (pergunta teórica)

Pergunta: "Como cobrar um condômino inadimplente?"
-> Sugerir: notificacao_inadimplencia
-> Mensagem: "Posso redigir uma Notificação de Inadimplência para você. Deseja que eu faça isso?"

Pergunta: "Redija uma notificação de barulho"
-> Não sugerir (já está pedindo explicitamente)"""

human_prompt = """Analise a seguinte pergunta e determine se é apropriado sugerir a redação de um documento:

PERGUNTA DO USUÁRIO:
{question}

RESPOSTA GERADA:
{answer}

Determine se deve sugerir um documento e qual tipo."""

suggest_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
])

document_suggester_chain = suggest_prompt | structured_output
