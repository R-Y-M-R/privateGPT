from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from chromadb.config import Settings

def main():
    # Disable telemetery in chroma settings
    persist_directory = 'db'
    chroma_settings = Settings(
	    chroma_db_impl='duckdb+parquet',
		persist_directory=persist_directory,
        anonymized_telemetry=False
    )
	
    # Load stored vectorstore
    llama = LlamaCppEmbeddings(model_path="./models/ggml-model-q4_0.bin")
    persist_directory = 'db'
    db = Chroma(persist_directory=persist_directory, embedding_function=llama, client_settings=chroma_settings)
    retriever = db.as_retriever()
    # Prepare the LLM
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model='./models/ggml-gpt4all-j-v1.3-groovy.bin', backend='gptj', callbacks=callbacks, verbose=False)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
        # Get the answer from the chain
        res = qa(query)    
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        
        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

if __name__ == "__main__":
    main()