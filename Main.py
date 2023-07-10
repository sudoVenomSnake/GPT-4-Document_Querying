import streamlit as st
import redirect as rd
from dotenv import load_dotenv

load_dotenv()

import os
import tempfile
import time

from llama_index import SimpleDirectoryReader, StorageContext, LLMPredictor
from llama_index import TreeIndex
from llama_index import ServiceContext
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain import LLMChain, OpenAI
from llama_index.indices.tree.tree_root_retriever import TreeRootRetriever
import re
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from llama_index.query_engine import RetrieverQueryEngine
import openai
from llama_index.tools import QueryEngineTool
from llama_index.query_engine import RouterQueryEngine
# import nest_asyncio

# nest_asyncio.apply()

os.environ["OPENAI_API_KEY"] = st.secrets("OPENAI_API_KEY")
openai.api_key = st.secrets("OPEN_AI_API")

query_engine_tools = []

import asyncio
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

def remove_formatting(output):
    output = re.sub('\[[0-9;m]+', '', output)  
    output = re.sub('\', '', output) 
    return output.strip()

@st.cache_resource
def preprocessing(uploaded_files):
    names = []
    descriptions = []
    if uploaded_files:
        temp_dir = tempfile.TemporaryDirectory()
        file_paths = []
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir.name, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            file_paths.append(file_path)
        
        for file_path in file_paths:
            document = SimpleDirectoryReader(input_files = [file_path]).load_data()
            index = TreeIndex.from_documents(document)
            engine = index.as_query_engine(similarity_top_k = 3)
            # print(engine.query('What are the problematic statements in this document?'))
            retriever = TreeRootRetriever(index)
            temp_engine = RetrieverQueryEngine(retriever = retriever)
            summary = temp_engine.query("Write a short concise summary of this document")
            heading = temp_engine.query("Write a short concise heading of this document")
            d = str(summary)
            n = str(heading)
            query_engine_tools.append(QueryEngineTool.from_defaults(
                query_engine = engine, 
                description = f"The document title is - {n}",
            ))
            names.append(n)
            descriptions.append(d)
        st.write(names)
        st.write(descriptions)
        
        global query_engine

        query_engine = RouterQueryEngine.from_defaults(
            query_engine_tools=[t for t in query_engine_tools]
        )


        return query_engine
    
@st.cache_resource
def run(_query_engine, query):
    if query:
        st.write(query_engine.query(query).response)
        return True

st.set_page_config(layout = "wide")

st.title("Document Querying using GPT-4")
st.write("Upload your files")

llm_predictor = LLMPredictor(llm = ChatOpenAI(temperature = 0, model_name = 'gpt-4', max_tokens = -1, openai_api_key = openai.api_key))

storage_context = StorageContext.from_defaults()
service_context = ServiceContext.from_defaults(llm_predictor = llm_predictor)

uploaded_files = st.file_uploader("Upload files", accept_multiple_files = True)
# name = st.text_input('Enter Agent Name or leave blank.')
# description = st.text_input('Enter Agent Description or leave blank.')

query_engine = preprocessing(uploaded_files)
ack = False

if query_engine:
    query = st.text_input('Enter your Query.', key = 'query_input')
    ack = run(query_engine, query)
    if ack:
        ack = False
        query = st.text_input('Enter your Query.', key = 'new_query_input')
        ack = run(query_engine, query)
        