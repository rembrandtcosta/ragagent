from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import MODEL_NAME

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


class DocumentRequest(BaseModel):
    is_explicit_request: bool = Field(
        description="Se o usuário está explicitamente pedindo para redigir/escrever um documento"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Tipo de documento: notificacao_barulho, notificacao_inadimplencia, advertencia, convocacao_assembleia, ata_assembleia, comunicado_geral"
    )
    document_name: Optional[str] = Field(
        default=None,
        description="Nome amigável do documento em português"
    )
    extracted_context: Optional[str] = Field(
        default=None,
        description="Informações contextuais extraídas da solicitação (ex: nome do vizinho, unidade, data, descrição do problema)"
    )


structured_output = llm.with_structured_output(DocumentRequest)

system_prompt = """Você é um detector de solicitações de documentos condominiais.

Sua tarefa é identificar se o usuário está EXPLICITAMENTE pedindo para redigir/escrever um documento.

PALAVRAS-CHAVE QUE INDICAM PEDIDO EXPLÍCITO:
- "redija", "redigir", "escreva", "escrever", "faça", "fazer", "elabore", "elaborar"
- "crie", "criar", "gere", "gerar", "monte", "montar"
- "preciso de um/uma", "quero um/uma", "me ajude a escrever"
- "modelo de", "template de"

TIPOS DE DOCUMENTOS:

1. notificacao_barulho - Notificação de Barulho
   - Palavras: barulho, ruído, som alto, música, festa, perturbação, sossego

2. notificacao_inadimplencia - Notificação de Inadimplência
   - Palavras: inadimplência, devedor, débito, atraso, cobrança, não pagou

3. advertencia - Advertência
   - Palavras: advertência, advertir, infração, descumprimento, violação

4. convocacao_assembleia - Convocação de Assembleia
   - Palavras: convocação, convocar, assembleia, reunião

5. ata_assembleia - Ata de Assembleia
   - Palavras: ata, registro, assembleia realizada

6. comunicado_geral - Comunicado Geral
   - Palavras: comunicado, aviso, informar, comunicar

EXEMPLOS:

"Redija uma notificação de barulho para o apartamento 101"
-> is_explicit_request: true, document_type: notificacao_barulho, extracted_context: "apartamento 101"

"Escreva uma advertência para o morador que estacionou na vaga errada"
-> is_explicit_request: true, document_type: advertencia, extracted_context: "estacionou na vaga errada"

"O que fazer quando o vizinho faz barulho?"
-> is_explicit_request: false (é uma pergunta, não um pedido de documento)

"Quais são as regras sobre barulho?"
-> is_explicit_request: false (pergunta informativa)

"Me ajude a escrever um comunicado sobre a obra no elevador"
-> is_explicit_request: true, document_type: comunicado_geral, extracted_context: "obra no elevador"

EXTRAIA O CONTEXTO:
Quando identificar um pedido explícito, extraia informações úteis como:
- Número da unidade/apartamento
- Descrição do problema
- Datas mencionadas
- Nomes mencionados
- Qualquer detalhe relevante para o documento"""

human_prompt = """Analise a seguinte mensagem e determine se é um pedido explícito para redigir um documento:

MENSAGEM DO USUÁRIO:
{question}

Identifique se é um pedido explícito, o tipo de documento e extraia o contexto relevante."""

detect_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
])

document_request_detector_chain = detect_prompt | structured_output
