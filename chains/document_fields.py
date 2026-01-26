from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import MODEL_NAME

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


class DocumentField(BaseModel):
    field_id: str = Field(description="Identificador único do campo (snake_case)")
    label: str = Field(description="Rótulo do campo para exibição ao usuário")
    field_type: str = Field(description="Tipo: text, textarea, date, number, select")
    required: bool = Field(description="Se o campo é obrigatório")
    placeholder: str = Field(description="Texto de exemplo para o campo")
    options: Optional[List[str]] = Field(default=None, description="Opções para campos select")


class DocumentFieldsResult(BaseModel):
    fields: List[DocumentField] = Field(description="Lista de campos necessários")


structured_output = llm.with_structured_output(DocumentFieldsResult)

system_prompt = """Você é um especialista em documentos condominiais brasileiros.

Sua tarefa é identificar quais informações são necessárias para preencher um documento específico.

CAMPOS COMUNS POR TIPO DE DOCUMENTO:

## NOTIFICAÇÃO DE BARULHO (notificacao_barulho)
- nome_condominio: Nome do condomínio
- unidade_infratora: Número da unidade/apartamento infrator
- nome_morador: Nome do morador (se conhecido)
- data_ocorrencia: Data da ocorrência
- hora_ocorrencia: Horário aproximado
- descricao_barulho: Tipo de barulho (música alta, festa, obras, etc.)
- nome_sindico: Nome do síndico

## NOTIFICAÇÃO DE INADIMPLÊNCIA (notificacao_inadimplencia)
- nome_condominio: Nome do condomínio
- unidade_devedora: Número da unidade devedora
- nome_proprietario: Nome do proprietário
- meses_devidos: Meses em atraso (ex: "janeiro, fevereiro e março de 2024")
- valor_total: Valor total do débito
- prazo_pagamento: Prazo para regularização (ex: "10 dias")
- nome_sindico: Nome do síndico

## ADVERTÊNCIA (advertencia)
- nome_condominio: Nome do condomínio
- unidade: Número da unidade
- nome_morador: Nome do morador
- descricao_infracao: Descrição detalhada da infração
- data_infracao: Data da infração
- artigo_regimento: Artigo do regimento violado (se conhecido)
- nome_sindico: Nome do síndico

## CONVOCAÇÃO DE ASSEMBLEIA (convocacao_assembleia)
- nome_condominio: Nome do condomínio
- tipo_assembleia: Tipo (Ordinária/Extraordinária)
- data_assembleia: Data da assembleia
- hora_assembleia: Horário
- local_assembleia: Local (salão de festas, área comum, etc.)
- pauta: Itens da pauta (lista)
- nome_sindico: Nome do síndico

## ATA DE ASSEMBLEIA (ata_assembleia)
- nome_condominio: Nome do condomínio
- data_assembleia: Data em que ocorreu
- hora_inicio: Horário de início
- hora_fim: Horário de término
- local: Local da assembleia
- numero_presentes: Número de unidades presentes
- pauta_deliberacoes: Resumo das deliberações
- nome_sindico: Nome do síndico

## COMUNICADO GERAL (comunicado_geral)
- nome_condominio: Nome do condomínio
- assunto: Assunto do comunicado
- mensagem: Conteúdo principal do comunicado
- data_vigencia: Data de início da vigência (se aplicável)
- nome_sindico: Nome do síndico

DIRETRIZES:
1. Retorne apenas os campos necessários para o tipo de documento
2. Use field_type apropriado (textarea para textos longos, date para datas)
3. Campos obrigatórios: nome_condominio, nome_sindico, e campos essenciais
4. Placeholders devem dar exemplos claros
5. Considere o contexto extraído da solicitação para pré-preencher campos"""

human_prompt = """Identifique os campos necessários para o seguinte documento:

TIPO DE DOCUMENTO: {document_type}
NOME DO DOCUMENTO: {document_name}

CONTEXTO DA SOLICITAÇÃO (informações já conhecidas):
{context}

Retorne a lista de campos que o usuário precisa preencher."""

fields_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
])

document_fields_chain = fields_prompt | structured_output


# Pre-defined fields for faster response (fallback)
DOCUMENT_FIELDS = {
    "notificacao_barulho": [
        {"field_id": "nome_condominio", "label": "Nome do Condomínio", "field_type": "text", "required": True, "placeholder": "Condomínio Residencial Solar"},
        {"field_id": "unidade_infratora", "label": "Unidade Infratora", "field_type": "text", "required": True, "placeholder": "Apartamento 101, Bloco A"},
        {"field_id": "nome_morador", "label": "Nome do Morador", "field_type": "text", "required": False, "placeholder": "João da Silva"},
        {"field_id": "data_ocorrencia", "label": "Data da Ocorrência", "field_type": "date", "required": True, "placeholder": ""},
        {"field_id": "hora_ocorrencia", "label": "Horário", "field_type": "text", "required": True, "placeholder": "23:30"},
        {"field_id": "descricao_barulho", "label": "Descrição do Barulho", "field_type": "textarea", "required": True, "placeholder": "Música alta, som de festa com muitas pessoas"},
        {"field_id": "nome_sindico", "label": "Nome do Síndico", "field_type": "text", "required": True, "placeholder": "Maria Santos"},
    ],
    "notificacao_inadimplencia": [
        {"field_id": "nome_condominio", "label": "Nome do Condomínio", "field_type": "text", "required": True, "placeholder": "Condomínio Residencial Solar"},
        {"field_id": "unidade_devedora", "label": "Unidade Devedora", "field_type": "text", "required": True, "placeholder": "Apartamento 205, Bloco B"},
        {"field_id": "nome_proprietario", "label": "Nome do Proprietário", "field_type": "text", "required": True, "placeholder": "José Pereira"},
        {"field_id": "meses_devidos", "label": "Meses em Atraso", "field_type": "text", "required": True, "placeholder": "Janeiro, Fevereiro e Março de 2024"},
        {"field_id": "valor_total", "label": "Valor Total do Débito", "field_type": "text", "required": True, "placeholder": "R$ 1.500,00"},
        {"field_id": "prazo_pagamento", "label": "Prazo para Pagamento", "field_type": "text", "required": True, "placeholder": "10 dias úteis"},
        {"field_id": "nome_sindico", "label": "Nome do Síndico", "field_type": "text", "required": True, "placeholder": "Maria Santos"},
    ],
    "advertencia": [
        {"field_id": "nome_condominio", "label": "Nome do Condomínio", "field_type": "text", "required": True, "placeholder": "Condomínio Residencial Solar"},
        {"field_id": "unidade", "label": "Unidade", "field_type": "text", "required": True, "placeholder": "Apartamento 302"},
        {"field_id": "nome_morador", "label": "Nome do Morador", "field_type": "text", "required": True, "placeholder": "Carlos Oliveira"},
        {"field_id": "data_infracao", "label": "Data da Infração", "field_type": "date", "required": True, "placeholder": ""},
        {"field_id": "descricao_infracao", "label": "Descrição da Infração", "field_type": "textarea", "required": True, "placeholder": "Descreva detalhadamente a infração cometida"},
        {"field_id": "artigo_regimento", "label": "Artigo do Regimento Violado", "field_type": "text", "required": False, "placeholder": "Art. 15, § 2º"},
        {"field_id": "nome_sindico", "label": "Nome do Síndico", "field_type": "text", "required": True, "placeholder": "Maria Santos"},
    ],
    "convocacao_assembleia": [
        {"field_id": "nome_condominio", "label": "Nome do Condomínio", "field_type": "text", "required": True, "placeholder": "Condomínio Residencial Solar"},
        {"field_id": "tipo_assembleia", "label": "Tipo de Assembleia", "field_type": "select", "required": True, "placeholder": "", "options": ["Ordinária", "Extraordinária"]},
        {"field_id": "data_assembleia", "label": "Data da Assembleia", "field_type": "date", "required": True, "placeholder": ""},
        {"field_id": "hora_assembleia", "label": "Horário", "field_type": "text", "required": True, "placeholder": "19:00"},
        {"field_id": "local_assembleia", "label": "Local", "field_type": "text", "required": True, "placeholder": "Salão de Festas"},
        {"field_id": "pauta", "label": "Pauta (itens separados por linha)", "field_type": "textarea", "required": True, "placeholder": "1. Aprovação das contas de 2023\n2. Eleição de síndico\n3. Assuntos gerais"},
        {"field_id": "nome_sindico", "label": "Nome do Síndico", "field_type": "text", "required": True, "placeholder": "Maria Santos"},
    ],
    "ata_assembleia": [
        {"field_id": "nome_condominio", "label": "Nome do Condomínio", "field_type": "text", "required": True, "placeholder": "Condomínio Residencial Solar"},
        {"field_id": "data_assembleia", "label": "Data da Assembleia", "field_type": "date", "required": True, "placeholder": ""},
        {"field_id": "hora_inicio", "label": "Horário de Início", "field_type": "text", "required": True, "placeholder": "19:00"},
        {"field_id": "hora_fim", "label": "Horário de Término", "field_type": "text", "required": True, "placeholder": "21:30"},
        {"field_id": "local", "label": "Local", "field_type": "text", "required": True, "placeholder": "Salão de Festas"},
        {"field_id": "numero_presentes", "label": "Unidades Presentes", "field_type": "number", "required": True, "placeholder": "25"},
        {"field_id": "pauta_deliberacoes", "label": "Deliberações", "field_type": "textarea", "required": True, "placeholder": "Descreva as deliberações tomadas em cada item da pauta"},
        {"field_id": "nome_sindico", "label": "Nome do Síndico", "field_type": "text", "required": True, "placeholder": "Maria Santos"},
    ],
    "comunicado_geral": [
        {"field_id": "nome_condominio", "label": "Nome do Condomínio", "field_type": "text", "required": True, "placeholder": "Condomínio Residencial Solar"},
        {"field_id": "assunto", "label": "Assunto", "field_type": "text", "required": True, "placeholder": "Manutenção dos Elevadores"},
        {"field_id": "mensagem", "label": "Mensagem", "field_type": "textarea", "required": True, "placeholder": "Informamos aos moradores que..."},
        {"field_id": "data_vigencia", "label": "Data de Vigência", "field_type": "date", "required": False, "placeholder": ""},
        {"field_id": "nome_sindico", "label": "Nome do Síndico", "field_type": "text", "required": True, "placeholder": "Maria Santos"},
    ],
}


def get_document_fields(document_type: str, document_name: str = "", context: str = "") -> list:
    """Get fields for a document type. Uses predefined fields for speed."""
    if document_type in DOCUMENT_FIELDS:
        return DOCUMENT_FIELDS[document_type]

    # Fallback to LLM for unknown types
    try:
        result = document_fields_chain.invoke({
            "document_type": document_type,
            "document_name": document_name,
            "context": context
        })
        return [field.dict() for field in result.fields]
    except Exception as e:
        print(f"Error getting document fields: {e}")
        # Return minimal fields
        return [
            {"field_id": "nome_condominio", "label": "Nome do Condomínio", "field_type": "text", "required": True, "placeholder": ""},
            {"field_id": "conteudo", "label": "Conteúdo", "field_type": "textarea", "required": True, "placeholder": ""},
            {"field_id": "nome_sindico", "label": "Nome do Síndico", "field_type": "text", "required": True, "placeholder": ""},
        ]
