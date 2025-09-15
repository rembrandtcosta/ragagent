from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_nomic import NomicEmbeddings

DOCS_PATH = "data"
loader = PyPDFDirectoryLoader(DOCS_PATH)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

splits = text_splitter.split_documents(docs)

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
