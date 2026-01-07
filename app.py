import streamlit as st
import pandas as pd 
import numpy as np
from typing import List
import langchain
from langchain_core.documents import Document
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chat_models.base import init_chat_model
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI

#Setup
st.set_page_config(page_title="Personal Financial Advisor")

# Initialize the OpenAI and Pinecone api keys in the environment (Note add the api keys in the .env file)
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"]=os.getenv("PINECONE_API_KEY")
pinecone_api_key=os.getenv("PINECONE_API_KEY")

if not os.environ["OPENAI_API_KEY"] or not pinecone_api_key:
    st.error("Missing OpenAI API Key or PineConeAPI Key. Check your .env file")
    st.stop()

@st.cache_resource
def build_chain():

    # Initializing Embedding, we are using openai embedding
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)

    #Initialize and connect to Vector Store and LLMs
    pc=Pinecone(api_key=pinecone_api_key)

    #Accessing the index
    index_name="personalfinance"
    index=pc.Index(index_name)

    # Initializing the Vector Store
    vector_store=PineconeVectorStore(index=index,embedding=embeddings)

    #Initialize Vector Store as Retriever
    retriever= vector_store.as_retriever()

    #Initialize LLM
    llm=ChatOpenAI(model="gpt-5.2",temperature=0.2, max_completion_tokens=500)

    ## create a prompt that includes the chat history
    contextualize_q_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt= ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}"),
    ])

    # Create a history aware retriever
    history_aware_retriever=create_history_aware_retriever(llm,retriever, contextualize_q_prompt)


    #Create a new document chain with history
    qa_system_prompt="""You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Context: {context}"""

    qa_prompt=ChatPromptTemplate.from_messages([
        ("system",qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ])

    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)

    Conversation_Rag_chain=create_retrieval_chain(history_aware_retriever,
                                                question_answer_chain)
    return Conversation_Rag_chain

rag=build_chain()

#Initialize Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

#Page UI 
st.title("Personal Finance Advisor")
st.caption("Ask Questions about your finances based on the Rag Knowledge base")

#Chat
for msg in st.session_state.chat_history:
    if isinstance(msg,HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg,AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

user_input=st.chat_input("Enter your Query")


if user_input:
    #Show user Message Immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    #Invoking RAG Chain
    result=rag.invoke({
        "chat_history": st.session_state.chat_history,
        "input":user_input
    })

    answer=result["answer"]

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))

    with st.chat_message("assistant"):
        st.markdown(answer)
