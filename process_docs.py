from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_nomic import NomicEmbeddings

DOCS_PATH = "data/codigo_civil_1314_1358"
loader = DirectoryLoader(path=DOCS_PATH, recursive=True, loader_cls=TextLoader)
docs = loader.load()

splits = docs

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=NomicEmbeddings(
        model="nomic-embed-text-v1.5",
        inference_mode="local"
    ),
    persist_directory="./db",
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    ),
)
