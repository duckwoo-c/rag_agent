from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI

import json

def load_docs_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)['glossary']
    # Assuming each entry in the JSON is a document
    return [Document(page_content=str(item)) for item in data]  # Adjust key as necessary
def load_docs_from_json2(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Assuming each entry in the JSON is a document
    return [Document(page_content=str(item)) for item in data]  # Adjust key as necessary

def embedding_model(api_info):
    embeddings = AzureOpenAIEmbeddings(
    openai_api_key=api_info["EMB_AZURE_API_KEY"],
    azure_endpoint=api_info["EMB_AZURE_ENDPOINT"],
    azure_deployment=api_info["EMB_AZURE_DEPLOYMENT"]
    )
    return embeddings

def create_llm(api_info, model_config):
    model = AzureChatOpenAI(
    azure_deployment=api_info["AZURE_DEPLOYMENT_NAME"],
    api_version=api_info["AZURE_API_VERSION"],
    temperature=model_config["temperature"],
    max_tokens=model_config["max_tokens"],
    timeout=model_config["timeout"],
    azure_endpoint=api_info["AZURE_ENDPOINT"],
    api_key=api_info["AZURE_API_KEY"]
    )
    return model

def docs_to_vectorstore(docs, name, embeddings, persist_dir = None):
    vectorstore = Chroma.from_documents(
    documents=docs,
    collection_name=name,
    embedding=embeddings,
    persist_directory=persist_dir
    )
    return vectorstore

def vectorstore_to_retriever_tool(vectorstore, tool_name, tool_desc):
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        tool_name,
        tool_desc
    )
    return retriever_tool
