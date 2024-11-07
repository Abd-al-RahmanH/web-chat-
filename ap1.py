import streamlit as st
from sqlalchemy import create_engine
from pymilvus import connections, Collection
import wikipediaapi
from dotenv import load_dotenv
import os
import requests
import numpy as np

# Load configuration
load_dotenv("config.env")

# Watsonx and Milvus configurations
MILVUS_HOST = os.getenv("MILVUS_HOST")
WATSONX_DB_URL = os.getenv("WATSONX_DB_URL")

# Set up Milvus connection
connections.connect("default", host=MILVUS_HOST, port="19530")

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('en')

st.title("Climate Change Q&A with RAG using Watsonx.ai and Milvus")

# Step 1: Data Collection and Insertion
st.header("1. Data Collection and Insertion")
topic = st.text_input("Enter topic (e.g., Climate Change):", "Climate Change")
if st.button("Fetch and Insert Wikipedia Data"):
    # 1. Fetch Wikipedia data
    def fetch_wikipedia_data(topic):
        page = wiki_wiki.page(topic)
        if page.exists():
            return page.text
        else:
            st.error("Wikipedia page not found.")
            return None

    # Retrieve the Wikipedia content
    content = fetch_wikipedia_data(topic)
    if content:
        # 2. Chunk the data
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]

        # 3. Insert into Watsonx.data
        def insert_into_watsonx(chunks):
            engine = create_engine(WATSONX_DB_URL)
            with engine.connect() as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS wiki_articles (id INT PRIMARY KEY, content TEXT)")
                for idx, chunk in enumerate(chunks):
                    conn.execute("INSERT INTO wiki_articles (id, content) VALUES (%s, %s)", (idx, chunk))
            st.success("Data inserted into Watsonx.data")

        insert_into_watsonx(chunks)

# Step 2: Embedding and Storing in Milvus
st.header("2. Embedding and Storing in Milvus")
if st.button("Create Embeddings and Insert into Milvus"):
    # Vectorize content and insert embeddings into Milvus
    def vectorize_text(text):
        # Placeholder for actual embedding model, e.g., SentenceTransformers
        return np.random.rand(1, 512)  # Example embedding, replace with actual embedding function

    # Define Milvus collection and insert data
    collection_name = "wiki_articles"
    collection = Collection(collection_name)

    # Prepare embeddings for Milvus insertion
    embeddings = [vectorize_text(chunk).tolist() for chunk in chunks]

    # Insert embeddings and original text into Milvus
    collection.insert(embeddings)
    st.success("Embeddings stored in Milvus")

# Step 3: Querying and Response Generation
st.header("3. Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    def query_milvus(query):
        # Placeholder function for querying Milvus and Watsonx
        # Query Milvus for similar embeddings
        results = collection.search([vectorize_text(query)], "content")
        return results  # replace with actual results processing

    def generate_response(results):
        # Placeholder for Watsonx.ai response generation using context
        return "Generated response based on Watsonx.ai"

    # Get answer and display
    milvus_results = query_milvus(question)
    answer = generate_response(milvus_results)
    st.write("Answer:", answer)
