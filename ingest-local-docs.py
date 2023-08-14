import os
import logging
import openai
import pinecone
import langchain
import nltk

from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader

from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

load_dotenv()

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)     

INDEX_NAME='pragya-teachus-bot'

index = pinecone.Index('pragya-teachus-bot')



def ingest_docs():
    """Ingests documents from the `docs` directory into Pinecone."""

    logging.info("Ingesting documents...")

    loader = DirectoryLoader("docs/")
    docs = loader.load()

    if not docs:
        logging.error("No documents found in the `docs` directory.")
        return

    print(f"loaded {len(docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=50)
    
    documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    print(f"Going to add {len(documents)} to Pinecone")

 
    Pinecone.from_documents(documents, embeddings, index_name='pragya-teachus-bot')
   

    logging.info("Done ingesting documents.")


if __name__ == "__main__":
    ingest_docs()


