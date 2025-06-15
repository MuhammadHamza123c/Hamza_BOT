from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st
import requests

api_key = st.secrets["api"]['grok_key']
url = "https://api.groq.com/openai/v1/chat/completions"
Headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

loader = TextLoader('hamza-port.txt')
document = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
chunks = splitter.split_documents(document)
embeddingss = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddingss)

user = st.text_input("User: ")
if user:
    query = user
    similar = vectorstore.similarity_search(query, k=3)
    context = [main.page_content for main in similar]
    prompt = f"""Your name is Hamza and when someone ask quesion to you which make sure you reply from Given Context. You are funny too!
Context: {context}
Query: {query}"""
    payload = {
        'model': 'llama3-8b-8192',
        'messages': [{
            'role': 'user', 'content': prompt
        }],
        'max_tokens': 60
    }
    response = requests.post(url=url, headers=Headers, json=payload)
    data = response.json()
    st.write(f'Hamza: {data["choices"][0]["message"]["content"]}')
