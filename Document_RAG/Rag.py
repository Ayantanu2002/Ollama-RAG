import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
import ollama

# Function to process the uploaded PDF
def parse_pdf(uploaded_file):
    docs = []
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text = page.get_text()
            docs.append(Document(page_content=text))  # Use the Document class here
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
def rag_chain(question, docs):
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
st.title("RAG System with Ollama and Local PDF")
st.write("Upload a PDF and enter a question to generate an answer based on the content of the file.")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Question input
question_input = st.text_input("Enter your question", "What is Python?")

# Add a submit button
submit_button = st.button(label="Submit")

# If the submit button is clicked, start the process
if submit_button:
    if uploaded_file and question_input:
        st.write("Processing PDF...")
        docs = parse_pdf(uploaded_file)
        st.write("Generating Answer...")
        result = rag_chain(question_input, docs)
        st.write("Answer:")
        st.write(result)
    else:
        st.write("Please upload a PDF file and provide a question.")
