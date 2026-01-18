import time
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# --- 1. CONFIGURAÇÃO ---
DOCS_PATH = "ingest/data/artigos"

# Instancia um LLM rápido para fazer a "tradução" para linguagem simples
# O gemini-1.5-flash é ótimo para isso: rápido e barato
llm_rewriter = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    temperature=0.3
)

# Template focado em alinhar o vocabulário jurídico com o do usuário leigo
rewrite_prompt = PromptTemplate.from_template(
    """Você é um especialista jurídico focado em traduzir "juridiquês" para linguagem popular.
    Analise o artigo de lei abaixo e explique sobre o que se trata o artigo. Use termos coloquiais se necessário.
    Responda com APENAS O TEXTO. Não inclua introduções ou conclusões.
    Somente o texto resumindo/reescrevendo o artigo.

    Artigo: "{text}"

    Texto explicando o artigo:"""
)

# --- 2. CARREGAMENTO ---
print("📂 Carregando documentos...")
loader = DirectoryLoader(path=DOCS_PATH, recursive=True, loader_cls=TextLoader)
docs = loader.load()

# --- 3. A MÁGICA: REESCRITA (DOCUMENT EXPANSION) ---
print(f"🔄 Iniciando reescrita de {len(docs)} documentos...")

enhanced_docs = []

# Criamos uma Chain simples para invocar o LLM
rewrite_chain = rewrite_prompt | llm_rewriter

for i, doc in enumerate(docs):
    original_text = doc.page_content
    
    # A) Preserva o original nos metadados (Backup de segurança)
    doc.metadata["original_content_pure"] = original_text
    
    try:
        # B) Gera a expansão (Linguagem Simples)
        # Invocamos o LLM. Adicionei um pequeno delay para evitar Rate Limit se tiver muitos docs
        simplification = rewrite_chain.invoke({"text": original_text}).content
        
        # C) A Estratégia Híbrida:
        # Colocamos a simplificação NO INÍCIO para o Cross-Encoder/Embeddings pegar primeiro.
        # Mantemos o TEXTO ORIGINAL depois, para o RAG poder citar a lei corretamente.
        new_content = (
            f"PALAVRAS-CHAVE E DÚVIDAS COMUNS:\n{simplification}\n\n"
            # f"TEXTO LEGAL ORIGINAL:\n{original_text}"
        )
        
        doc.page_content = new_content
        print(f"   ✅ Doc {i+1}/{len(docs)} processado.")
        
    except Exception as e:
        print(f"   ❌ Erro ao processar doc {i+1}: {e}")
        # Se der erro, mantém o original para não perder o dado
        doc.page_content = original_text
    
    # Delay de segurança para API gratuita (opcional)
    # time.sleep(1) 
    enhanced_docs.append(doc)

# --- 4. INDEXAÇÃO (VECTOR STORE) ---
print("💾 Salvando no ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=enhanced_docs, # Usamos a lista enriquecida
    embedding=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="retrieval_document"
    ),
    persist_directory="./db",
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    ),
)
print("🚀 Concluído!")
