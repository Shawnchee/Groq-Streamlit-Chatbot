import os

import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

load_dotenv()

GROQ_API_KEY = os.environ['GROQ_API_KEY']

working_dir = os.path.dirname(os.path.abspath(__file__))

poppler_path = r'C:\Users\aykay\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin'

def load_document(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
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


st.set_page_config(
    page_title="Chat with Document",
    page_icon="🔗",
    layout="centered"
    
)

st.title("Chat with Document")

# initialize the chat history in streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "vectorstores" not in st.session_state:
        st.session_state.vectorstores = setup_vectorstore(load_document(file_path))

    if "conversational_chain" not in st.session_state:
        st.session_state.conversational_chain = create_chain(st.session_state.vectorstores)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question") 

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversational_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})












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


