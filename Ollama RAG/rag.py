import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
import ollama

# Function to fetch and parse the webpage
def fetch_and_parse(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    content = soup.find_all(['h1', 'h2', 'p'])  # Extract headings and paragraphs
    docs = []
    for tag in content:
        docs.append(Document(page_content=tag.get_text()))  # Use the Document class here
    return docs

# Function to format retrieved docs
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

# Function to query the LLaMA model using Ollama
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": formatted_prompt}])
    return response['message']['content']

# Define the RAG Chain
def rag_chain(question, url):
    # Fetch the content from the URL
    docs = fetch_and_parse(url)

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)

    # Create embeddings using Ollama
    embeddings = OllamaEmbeddings(model="mistral")

    # Create vector store from documents and embeddings
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever()

    # Retrieve the relevant documents for the question
    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)

    # Generate answer using LLaMA model
    return ollama_llm(question, context)

# Streamlit UI
st.title("RAG System with Ollama and Wikipedia")
st.write("Enter a URL and a question to generate an answer based on the content of the page.")

# URL input
url_input = st.text_input("Enter URL", "https://en.wikipedia.org/wiki/Python_(programming_language)")

# Question input
question_input = st.text_input("Enter your question", "What is Python?")

# Add a submit button
submit_button = st.button(label="Submit")

# If the submit button is clicked, start the process
if submit_button:
    if url_input and question_input:
        st.write("Generating Answer...")
        result = rag_chain(question_input, url_input)
        st.write("Answer:")
        st.write(result)
    else:
        st.write("Please provide both a URL and a question.")
