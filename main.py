import os
import streamlit as st
import time
import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

# Access environment variables
api_key = os.getenv("API_KEY")


import asyncio
import nest_asyncio

st.title("AI-POWERED REASEARCH TOOL 📈")

st.sidebar.title('News Articl Urls')

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i}")
    urls.append(url)

process_url_clicked = st.sidebar.button('Process Urls')


main_placefolder = st.empty()

   

nest_asyncio.apply()

try:
        asyncio.get_running_loop()
except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


# initialize ll
llm = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    temperature = 0.9,
    google_api_key = api_key
)

if process_url_clicked:
    #load data
	loader = UnstructuredURLLoader(urls = urls)
	main_placefolder.text('Data Loading Started...')
	data = loader.load()
	
	# split data
	text_splitter = RecursiveCharacterTextSplitter(
		separators = ['\n\n','\n', '.' , ','],
		chunk_size = 1000,
		chunk_overlap = 200
	)

	main_placefolder.text('Text Splitter Started...')
	docs = text_splitter.split_documents(data)
	

	# embeddings and save it to FAISS index
	embeddings = GoogleGenerativeAIEmbeddings(
		model = "models/embedding-001",
    	google_api_key = api_key 
	)
	# vector db

	vector_store_gemini = FAISS.from_documents(docs,embeddings)
	main_placefolder.text('Embedding Vector Started Building....')	

	time.sleep(2)	

	# save 
	vector_store_gemini.save_local('vector_db_index')
       

query = main_placefolder.text_input("Question: ")

if query:
    vector_index = FAISS.load_local("vector_db_index", embeddings, allow_dangerous_deserialization=True)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_index.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)
    # result will be a dictionary of this format --> {"answer": "", "sources": [] }
    st.header("Answer")
    st.write(result["answer"])

    # Display sources, if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)


