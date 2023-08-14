import os
import pinecone
from dotenv import load_dotenv
from apify_client import ApifyClient
from langchain.vectorstores import Pinecone
from langchain.document_loaders import ApifyDatasetLoader, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

# Define the index name
INDEX_NAME = 'pragya-teachus-bot'

def ingest_docs():
    """
    Ingests documents from the `docs` directory and URLs into Pinecone.
    """

    print("Ingesting documents...")

    # Initialize ApifyClient with API token
    apify_client = ApifyClient(os.environ["APIFY_API_TOKEN"])

    # Run Apify actor for website content crawling
    actor_call = apify_client.actor("apify/website-content-crawler").call(
        run_input={"startUrls": [{"url": "https://www.pima.edu/"}]},
    )

    # Define a function to map Apify dataset items to Langchain documents
    def map_dataset_item(item):
        return Document(page_content=item["text"] or "", metadata={"source": item["url"]})

    # Initialize ApifyDatasetLoader using the obtained dataset ID and mapping function
    url_loader = ApifyDatasetLoader(
        dataset_id=actor_call["defaultDatasetId"],
        dataset_mapping_function=map_dataset_item,
    )

    # Load documents from URLs
    docs = url_loader.load()

    print(f"Loaded {len(docs)} documents.")

    # Initialize RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=50
    )

    # Split documents using the text splitter
    documents = text_splitter.split_documents(docs)
    print(f"Split into {len(documents)} documents.")

    # Initialize OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    # Ingest documents into Pinecone
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)

    print("Done ingesting documents.")

if __name__ == "__main__":
    ingest_docs()
