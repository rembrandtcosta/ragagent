from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

DOCS_PATH = "data/artigos"
loader = DirectoryLoader(path=DOCS_PATH, recursive=True, loader_cls=TextLoader)
docs = loader.load()
print(docs)
splits = docs

vectorstore = Chroma.from_documents(
    documents=splits,
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
