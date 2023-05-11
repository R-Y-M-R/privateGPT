from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.embeddings import LlamaCppEmbeddings
from sys import argv

def main():
	# Disable telemetery in chroma settings
    persist_directory = 'db'
    chroma_settings = Settings(
	    chroma_db_impl='duckdb+parquet',
		persist_directory=persist_directory,
        anonymized_telemetry=False
    )
	
    # Load document and split in chunks
    loader = TextLoader(argv[1], encoding="ISO-8859-1") 
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    # Create embeddings
    llama = LlamaCppEmbeddings(model_path="./models/ggml-model-q4_0.bin")
    # Create and store locally vectorstore
    
    db = Chroma.from_documents(texts, llama, persist_directory=persist_directory, client_settings=chroma_settings)
    db.persist()
    db = None

if __name__ == "__main__":
    main()