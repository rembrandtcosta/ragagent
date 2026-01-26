from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import MODEL_NAME

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


class IllegalityAnalysis(BaseModel):
    is_potentially_illegal: bool = Field(
        description="Se a cláusula é potencialmente ilegal"
    )
    confidence: str = Field(
        description="Nível de confiança: 'alta', 'media', 'baixa'"
    )
    conflicting_articles: List[str] = Field(
        description="Lista de artigos do Código Civil que podem ser violados"
    )
    explanation: str = Field(
        description="Explicação detalhada do potencial conflito legal"
    )
    legal_principle_violated: Optional[str] = Field(
        default=None,
        description="Princípio jurídico violado (ex: direito de propriedade, proporcionalidade)"
    )
    recommendation: str = Field(
        description="Recomendação de ação para o condomínio"
    )


structured_output = llm.with_structured_output(IllegalityAnalysis)

system_prompt = """Você é um advogado especialista em direito condominial brasileiro, com profundo conhecimento do Código Civil.

Sua tarefa é analisar cláusulas de convenções e regimentos internos de condomínio para identificar potenciais ilegalidades.

PADRÕES DE ILEGALIDADE CONHECIDOS:

1. PROIBIÇÃO TOTAL DE ANIMAIS DE ESTIMAÇÃO
   - Viola: Art. 1228 (direito de propriedade), Art. 1336 IV (apenas "mau uso" pode ser restringido)
   - Princípio: Direito de propriedade, função social
   - A convenção pode apenas regulamentar, não proibir totalmente

2. RESTRIÇÃO DE ÁREAS COMUNS PARA INADIMPLENTES
   - Viola: Art. 1335 II (direito de usar áreas comuns)
   - Princípio: Vedação de sanção política
   - Condômino inadimplente mantém direito de uso das áreas comuns

3. MULTAS EXCESSIVAS
   - Multa por infração > 5x a taxa condominial: viola Art. 1336 §2º
   - Multa por comportamento antissocial > 10x: viola Art. 1337
   - Princípio: Proporcionalidade

4. QUORUM ABAIXO DO MÍNIMO LEGAL
   - Quorum para alteração de convenção < 2/3: viola Art. 1351
   - Quorum para obras voluptuárias < 2/3: viola Art. 1341 I
   - Quorum para obras úteis < maioria: viola Art. 1341 II

5. PROIBIÇÃO DE VISITANTES/HÓSPEDES
   - Viola: Art. 1228 (direito de propriedade)
   - Princípio: Livre uso da propriedade
   - Pode regulamentar, não proibir

6. RESTRIÇÕES IRRAZOÁVEIS AO USO DA PROPRIEDADE
   - Viola: Art. 1228, Art. 1336 IV
   - Princípio: Proporcionalidade, razoabilidade
   - Restrições devem ser justificadas e proporcionais

ARTIGOS RELEVANTES DO CÓDIGO CIVIL:

{relevant_articles}

DIRETRIZES DE ANÁLISE:

1. Analise a cláusula objetivamente
2. Identifique se há conflito com a legislação
3. Considere jurisprudência consolidada
4. Seja preciso na identificação dos artigos violados
5. Forneça explicação clara e fundamentada
6. Recomende ações práticas

NÍVEIS DE CONFIANÇA:
- alta: conflito claro e direto com a lei
- media: conflito provável, mas depende de interpretação
- baixa: possível conflito, mas não pacificado"""

human_prompt = """Analise a seguinte cláusula de documento condominial quanto à sua legalidade:

CLÁUSULA:
Número: {clause_number}
Tópico: {clause_topic}
Texto: {clause_text}

Determine se esta cláusula é potencialmente ilegal, identifique os artigos do Código Civil que podem estar sendo violados, e forneça uma explicação detalhada e recomendação."""

analyze_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
])

illegality_detector_chain = analyze_prompt | structured_output
