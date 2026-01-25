from typing import List, Optional
from config import MODEL_NAME
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


class ExtractedClause(BaseModel):
    clause_number: str = Field(
        description="Identificador da cláusula (ex: 'Art. 15', 'Parágrafo 3º', 'Item 2.1')"
    )
    clause_text: str = Field(
        description="Texto completo da cláusula"
    )
    topic: str = Field(
        description="Tópico da cláusula: pets, common_areas, fees, quorum, visitors, property_use, fines, general"
    )


class ClauseExtractionResult(BaseModel):
    clauses: List[ExtractedClause] = Field(
        description="Lista de cláusulas extraídas do documento"
    )


structured_output = llm.with_structured_output(ClauseExtractionResult)

system_prompt = """Você é um especialista em análise de documentos condominiais brasileiros (convenção de condomínio e regimento interno).

Sua tarefa é extrair cláusulas individuais do texto fornecido.

DIRETRIZES PARA EXTRAÇÃO:

1. IDENTIFICAÇÃO DE CLÁUSULAS:
   - Identifique artigos, parágrafos, incisos e itens numerados
   - Cada cláusula deve ser uma unidade normativa completa
   - Preserve a numeração original (Art. 1º, § 2º, inciso III, etc.)

2. CLASSIFICAÇÃO POR TÓPICO:
   - pets: regras sobre animais de estimação
   - common_areas: uso de áreas comuns (piscina, salão, academia, etc.)
   - fees: taxas condominiais, multas, cobrança
   - quorum: votação, assembleias, deliberações
   - visitors: regras sobre visitantes e hóspedes
   - property_use: uso da unidade, reformas, barulho
   - fines: penalidades, multas, sanções
   - general: outras regras que não se encaixam nas categorias acima

3. REQUISITOS:
   - Extraia apenas cláusulas completas e bem definidas
   - Não fragmente cláusulas que devem permanecer juntas
   - Identifique corretamente o número/identificador de cada cláusula
   - Classifique o tópico com base no conteúdo principal da cláusula"""

human_prompt = """Analise o seguinte trecho de documento condominial e extraia todas as cláusulas presentes:

TEXTO DO DOCUMENTO:
{document_chunk}

Extraia cada cláusula com seu identificador, texto completo e tópico."""

extract_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
])

clause_extractor_chain = extract_prompt | structured_output


def deduplicate_clauses(clauses: List[ExtractedClause]) -> List[ExtractedClause]:
    """Remove duplicate clauses based on clause_number."""
    seen = set()
    unique_clauses = []
    for clause in clauses:
        if clause.clause_number not in seen:
            seen.add(clause.clause_number)
            unique_clauses.append(clause)
    return unique_clauses
