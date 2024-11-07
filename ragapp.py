from langchain.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
from watsonxlangchain import LangChainInterface

# Configuration

# Model
selected_model = 'ibm/granite-13b-chat-v2'

# Resources
resource_name = 'About Us - ProfitOptics.html'
pdf_resource_name = 'buhari.pdf'  # Replace with your PDF file path

## End configuration

creds = {
    'apikey':'CycH3S8_zauKHDxJvjtenKAkOnz6skxApg9VMECFyvX8', 
    'url': 'https://us-south.ml.cloud.ibm.com'
}

llm = LangChainInterface(
    credentials = creds, 
    model = selected_model, 
    params = {'decoding_method':'sample', 'max_new_tokens':200, 'temperature':0.5}, 
    project_id='5bd59c57-bfa7-4565-b392-a98a90224509')

@st.cache_resource
def load_external_resource(): 

    loaders = [
        UnstructuredHTMLLoader(resource_name),
        PyPDFLoader(pdf_resource_name)  # Add the PDF loader here
    ]

    index = VectorstoreIndexCreator(
        embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'), 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)

    return index

index = load_external_resource()

chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=index.vectorstore.as_retriever(), input_key='question')

st.title('Ask watsonx.ai 🤖')

if 'messages' not in st.session_state: 
    st.session_state.messages = [] 

for message in st.session_state.messages: 
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Pass Your Prompt here')

if prompt: 
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})

    response = chain.run(prompt)

    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})
