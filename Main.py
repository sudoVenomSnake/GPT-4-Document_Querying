import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import os
import tempfile
from llama_index import SimpleDirectoryReader, StorageContext, LLMPredictor
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from langchain.chat_models import ChatOpenAI
import tiktoken
import openai

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

@st.cache_resource
def preprocessing(uploaded_file):
    if uploaded_file:
        temp_dir = tempfile.TemporaryDirectory()
        file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        document = SimpleDirectoryReader(input_files = [file_path]).load_data()
        tokens = num_tokens_from_string(document[0].text, 'gpt-4')
        global context
        context = document[0].text
        if tokens <= 8000:
            print('Case - A')
            return context
        else:
            print('Case - B')
            index = index = VectorStoreIndex.from_documents(document)
            global engine
            engine = index.as_query_engine(similarity_top_k = 3)
            return engine

@st.cache_resource
def run(_query_engine, query):
    if type(_query_engine) == str:
        print('Executing Case - A')
        response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                        {"role": "system", "content": "You are a helpful assistant who answers questions given context."},
                        {"role": "user", "content": f"The question is - {query}\nThe provided context is - {_query_engine}\nAnswer the question to the best of your abilities."},
                    ]
                )
        st.write(response['choices'][0]['message']['content'])
    else:
        print('Executing Case - B')
        st.write(query_engine.query(query).response)
        return True

st.set_page_config(layout = "wide")

st.title("Document Querying using GPT-4")

llm_predictor = LLMPredictor(llm = ChatOpenAI(temperature = 0, model_name = 'gpt-4', max_tokens = -1, openai_api_key = openai.api_key))

storage_context = StorageContext.from_defaults()
service_context = ServiceContext.from_defaults(llm_predictor = llm_predictor)

uploaded_file = st.file_uploader('Upload your file')

query_engine = preprocessing(uploaded_file)

if query_engine:
    query = st.text_input('Enter your Query.', key = 'query_input')
    if query:
        run(query_engine, query)