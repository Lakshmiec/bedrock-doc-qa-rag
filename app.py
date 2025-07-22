import json
import os
import sys
import boto3
import streamlit as st

# Importing necessary libraries for LangChain and Bedrock for Embeddings and LLMs
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

# Data Ingestion : Importing necessary libraries for PDF processing    
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Importing necessary libraries for FAISS vector store
from langchain_community.vectorstores import FAISS

# Importing necessary libraries for document retrieval and QA
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Initialize the Bedrock client for LLMs
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(
    client=bedrock,
    model_id="amazon.titan-embed-text-v2:0")

# Data Ingestion: Function to load and process PDF files
def load_pdf_files(directory):
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    
    # Splitting the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    
    return split_docs


# Function that embeds the documents and saves the vector store locally
def create_faiss_vector_store(documents):
    # embeddings = bedrock_embeddings.embed_documents(documents)
    vector_store = FAISS.from_documents(documents, bedrock_embeddings)
  
    #another way to create vector store
    #vectorstore_faiss=FAISS.from_documents(   docs, bedrock_embeddings )

    vector_store.save_local("faiss_index")

# Function to load the LLM from Bedrock through LangChain
def get_llm():
    """Initialize and return the Bedrock LLM. Here we dont need to use the invoke_model method directly, as LangChain handles it. We just need to set up the LLM with the Bedrock client. and model ID. call it."""
    # You can find the model arguments for respective LLMs in the AWS console or documentation.
    llm_response = Bedrock(
        client=bedrock,
        model_id="us.deepseek.r1-v1:0", model_kwargs={"max_tokens": 512}
    )
    return llm_response

# Prompt template for the question-answering chain

prompt_template = """
You are a helpful assistant.Answer the question based on the context provided. Do not make up answers or provide information that is not in the context. 

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

def get_llm_response(vectore_store_faiss, query):
    llm = get_llm()
    # Create a RetrievalQA chain with the LLM and the vector store retriever
    # The retriever will use the FAISS vector store to find relevant documents based on the query
    # The chain_type "stuff" means it will concatenate the retrieved documents and pass them
    # to the LLM for generating an answer.
    # The return_source_documents=True will return the documents used to generate the answer.
    # The chain_type_kwargs={"prompt": PROMPT} will use the custom prompt template defined
    # above to format the input for the LLM.
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectore_store_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    response = qa({"query": query})
    return response

# Streamlit app to interact with the user
def main():
    st.set_page_config("Chat PDF")
  
    st.header("Chat with your PDFs using AWS Bedrock 💬")

    with st.sidebar:
        st.subheader("Create Vector Store")
        if st.button("Vector Store Update"):
            with st.spinner("Creating vector store..."):
            # This button will trigger the creation of the FAISS vector store
                # Load PDF files from the specified directory
                # Ensure the directory path is correct for your environment
                pdf_directory = "Data"
                if not os.path.exists(pdf_directory):
                    st.error(f"Directory {pdf_directory} does not exist.")
                else:
                    documents = load_pdf_files(pdf_directory)
                    if documents:
                        create_faiss_vector_store(documents)
                        st.success("Vector store created successfully!")
                    else:
                        st.error("No documents found in the specified directory.")

    user_query = st.text_input("Ask your question about the documents:")
    if user_query and st.button("Get Answer"):
        # Load the FAISS vector store
        
        vector_store_faiss = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
        # Get the LLM response based on the user query and the vector store
        response = get_llm_response(vector_store_faiss, user_query)
        st.write("Answer:")
        st.write(response['result'])
        st.write("Source Documents:")
        for doc in response['source_documents']:
            st.write(doc.page_content)
       
        st.success("Answer retrieved successfully!")

   
if __name__ == "__main__":
    main()
