import os
from flask import request
from pinecone import Pinecone, ServerlessSpec, PodSpec
import time
from langchain_openai import OpenAIEmbeddings 
from unstructured.partition.api import partition_via_api
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from pydantic import BaseModel
from typing import Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
import tempfile
import requests

pinecone_api_key = os.getenv('PINECONE_API_KEY')   # Get the Pinecone API key from the environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')      # Get the OpenAI API key from the environment variables
unstructured_api_key = os.getenv('UNSTRUCTURED_API_KEY')  # Get the Unstructured API key from the environment variables

s = UnstructuredClient(api_key_auth=unstructured_api_key)  # Initialize the Unstructured client

use_serverless = True

pc = Pinecone(api_key=pinecone_api_key)                # Initialize Pinecone client


def index_init(database):
    """
    Initialize the index for the specified database.

    Args:
        database (str): The name of the database.

    Returns:
        None
    """
    if use_serverless:
        spec = ServerlessSpec(cloud='aws', region='us-east-1')
    else:
        spec = PodSpec(size='medium')

    index_name = database
    if index_name not in pc.list_indexes().names():
        pc.create_index(index_name, dimension=1536, spec=spec, metric='cosine')  # Create a new index if it doesn't exist

    while not pc.describe_index(index_name).status['ready']:  
        time.sleep(1)

    ## Define Embeddings model 
    # get openai api key from platform.openai.com  

def embed_and_store(table_element, table_summary, index_name):
    """Embeds content and stores in Pinecone with metadata"""
    model_name = 'text-embedding-3-small'  
    embeddings = OpenAIEmbeddings(  
        model=model_name,  
        openai_api_key=openai_api_key  
    )
    text_field = "text"
    index = pc.Index(index_name)  
    vectorstore = PineconeVectorStore(  
        index, embeddings, text_field  
        )
    
    if table_element.category == "Table":
        var = table_element.metadata.text_as_html
    else:
        var = table_element.text
    meta = {
        "content_type": table_element.category,  # 'text' or 'table'
        "raw_content": var,
        "filename": table_element.metadata.filename,
        "page_number": table_element.metadata.page_number
    }

    # meta[self._text_key] = table_summary
    vectorstore.add_texts([table_summary], metadatas=[meta])

def partition_doc(file_path, index):
    elements = partition_via_api(
        file_path,
        strategy="hi_res",
        hi_res_model_name='yolox',
        infer_table_structure=True,
        chunking_strategy='by_title',
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )

    raw_pdf_elements = elements

# Create a dictionary to store counts of each type
    category_counts = {}

    for element in raw_pdf_elements:
        category = str(type(element))
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1

    class Element(BaseModel):
        type: str
        text: Any

    tables = [el for el in elements if el.category == "Table"]
    text = [el for el in elements if el.category == "CompositeElement"]
    
    prompt_text = """You are an assistant tasked with summarizing tables and text. \ 
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openai_api_key)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Apply to tables
    tables_text = [i.text for i in tables]
    table_summaries = summarize_chain.batch(tables_text, {"max_concurrency": 5})

    for i, s in enumerate(tables):
        print(f"Table {i}: {s.metadata.text_as_html}")
        embed_and_store(s, table_summaries[i], index)

    for i in text:
        print(f"Summarizing Text: {i.text}")
        i.summary = summarize_chain.invoke(i.text)

    for i in text:
        print(f"Uploading Summary: {i.text}")
        embed_and_store(i, i.summary, index)

def doc_upload():
    files = request.files.getlist("file")
    print(files)
    database = request.form.get("database")
    print(database)
    results = []

    index_init(database)

    for file in files:
        with tempfile.TemporaryDirectory() as temp_dir:   
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                file.save(file_path)
                print(f'Processing file: {file.filename}')
                partition_doc(file_path, database)
                results.append(file.filename)
                gateway_url = 'https://ffrgmk2r1k.execute-api.ca-central-1.amazonaws.com/addDocToDatabase'

                params = {
                    'database': database,
                    'document': file.filename
                }

                response = requests.get(gateway_url, params=params)

                if response.status_code == 200:
                    return (response.json()), 200
                else:
                    return ({'error': 'Failed to add document'}), response.status_code