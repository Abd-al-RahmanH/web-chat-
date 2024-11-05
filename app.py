# app.py
import os
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import spacy
import streamlit as st
from sentence_transformers import SentenceTransformer
from chromadb import Client, Settings
from chromadb.api.types import EmbeddingFunction
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

# Load API key and Project ID from .env file
load_dotenv(".env")
API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")

# Constants for the application
MODEL_TYPE = "codellama/codellama-34b-instruct-hf"
CACHE_DIR = ".cache"
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
url = "https://us-south.ml.cloud.ibm.com"

# Initialize the transformer model
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)
nlp = spacy.load("en_core_web_md")

# Set up ChromaDB client with cache settings
settings = Settings(persist_directory=CACHE_DIR)
client = Client(settings)

# Custom Embedding Function using MiniLM
class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
    MODEL = model

    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()

def extract_text(url):
    """Fetch and clean text content from a website."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = " ".join(p.get_text() for p in soup.find_all('p')).replace("\xa0", " ")
            return text
    except Exception as e:
        st.error(f"Error fetching content from {url}: {e}")
    return ""

def split_text_into_sentences(text):
    """Split text into sentences using SpaCy NLP."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def create_collection_name(url):
    """Create a collection name based on the domain of the URL."""
    return urlparse(url).netloc.split('.')[-2]

def create_embeddings(text, collection_name):
    """Create and upload embeddings to ChromaDB."""
    sentences = split_text_into_sentences(text)
    collection = client.get_or_create_collection(collection_name)
    collection.upsert(
        documents=sentences,
        metadatas=[{"source": str(i)} for i in range(len(sentences))],
        ids=[str(i) for i in range(len(sentences))]
    )
    return collection

def query_collection(collection, question):
    """Query ChromaDB collection for relevant context."""
    relevant_chunks = collection.query(query_texts=[question], n_results=5)
    return "\n\n".join(relevant_chunks["documents"][0])

def get_model():
    """Set up and return the Watson model."""
    generate_params = {
        GenParams.MAX_NEW_TOKENS: 100,
        GenParams.MIN_NEW_TOKENS: 50,
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.TEMPERATURE: 0.7,
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 1
    }
    return Model(
        model_id=MODEL_TYPE,
        params=generate_params,
        credentials={"apikey": API_KEY, "url": url},
        project_id=PROJECT_ID
    )

def generate_answer(model, context, question):
    """Create a prompt and use the model to generate an answer."""
    prompt = (
        f"### Context:\n{context}\n\n### Instruction:\nPlease answer concisely.\n"
        f"Question: {question}\nAnswer:"
    )
    response = model.generate(prompt=prompt)
    return response['results'][0]['generated_text'].strip()

def main():
    """Run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="IBM Watson WebChat")
    st.title("IBM Watsonx.ai - Web Q&A")

    st.sidebar.title("Settings")
    st.sidebar.markdown("Credentials are loaded from environment variables.")

    # Main input fields
    user_url = st.text_input("Provide a Website URL", "")
    question = st.text_area("Question", height=100)
    if st.button("Get Answer") and user_url and question:
        with st.spinner("Processing..."):
            text = extract_text(user_url)
            if text:
                collection_name = create_collection_name(user_url)
                collection = create_embeddings(text, collection_name)
                context = query_collection(collection, question)
                model = get_model()
                answer = generate_answer(model, context, question)
                st.subheader("Response")
                st.write(answer)
            else:
                st.warning("Could not extract text from the URL provided.")

    # Option to clear database memory
    if st.sidebar.button("Clear Memory"):
        client.get_collection(create_collection_name(user_url)).delete()
        st.sidebar.success("Memory cleared successfully!")

if __name__ == "__main__":
    main()
