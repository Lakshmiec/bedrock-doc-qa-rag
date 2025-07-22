# Bedrock DocQA RAG

A Document Q&A application built with **LangChain** and **AWS Bedrock**, using **Retrieval-Augmented Generation (RAG)** to answer questions from PDF files.

## 🚀 Features

- Query multiple PDFs from the memory
- Vector-based semantic search
- Use Bedrock-supported models: Deepseek, Amazon Titan
- Model selector UI (Streamlit )
- Configurable chunking, embedding, and response generation

## Steps
The project consists of several main steps:

- Data Ingestion: Read all PDFs from a folder and split them into chunks.
- Vector Embedding: Convert document chunks into embeddings using Amazon Titan or other embedding models.
- Vector Store: Store embeddings in a vector database (e.g., FAISS).
- Retrieval and LLM Integration: On query, perform a similarity search, retrieve relevant chunks, and use an LLM to generate an answer.



<img width="1424" height="588" alt="image" src="https://github.com/user-attachments/assets/9a4f8e01-7247-4767-b7ec-44c89812544b" />

## 🔧 Setup

```bash
git clone https://github.com/Lakshmiec/bedrock-doc-qa-rag.git
cd bedrock-doc-qa-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


