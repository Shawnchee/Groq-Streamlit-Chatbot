import os

import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

load_dotenv()

GROQ_API_KEY = os.environ['GROQ_API_KEY']

working_dir = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n",
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

def create_chain(vectorstore):
    llm = ChatGroq(
        model= "llama-3.1-70b-versatile",
        temperature=0.1
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm = llm,
        output_key= "answer",
        memory_key= "chat_history",
        return_messages= True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= retriever,
        memory= memory,
        verbose= True
    )
    return chain











# client = Groq(
#     api_key = GROQ_API_KEY
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Who drew the Monalisa Portrait?"
#         }
#     ],
#     model="llama3-8b-8192",
#     max_tokens=3
# )

# print(chat_completion.choices[0].message.content)


