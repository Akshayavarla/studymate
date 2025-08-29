from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_document(file):
    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(file.name)
    else:
        loader = TextLoader(file.name, encoding="utf-8")
    return loader.load()

def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings()
    return FAISS.from_documents(chunks, embeddings)
