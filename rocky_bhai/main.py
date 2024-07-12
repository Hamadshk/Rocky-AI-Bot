from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import os
import streamlit as st
import pickle
import time
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm, trange
from dotenv import load_dotenv
load_dotenv()

file_path="store_vec_index.pkl"

urls=[]

main_placefolder=st.empty()
chat = ChatGroq(
    temperature=1,            # Set the temperature for response generation
    model="llama3-70b-8192",  # Specify the GROQ model to use
    api_key="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"  # Optional: API key if not set as environment variable
)

st.title("News Research tool ")
st.sidebar.title("News Article URLS")

for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Process URLs")

if process_url_clicked:
    loader=UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data is LOading")
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000
    )
    main_placefolder.text("Data is splitting")

    docs=text_splitter.split_documents(data)
    embeddings=HuggingFaceEmbeddings()

    main_placefolder.text("Embedding vec started building ")

    index=FAISS.from_documents(docs,embeddings)
    with open(file_path,"wb") as f:
        pickle.dump(index,f)
query=main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
      with open(file_path,"rb") as f:
         inde=pickle.load(f)
         chain=RetrievalQAWithSourcesChain.from_llm(llm=chat,retriever=inde.as_retriever())
         result=chain({"question":query },return_only_outputs=True)
         st.header("Answer")
         st.subheader(result["answer"])



