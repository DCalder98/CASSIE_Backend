import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=openai_api_key)

query_expansion_prompt = PromptTemplate(
    input_variables=["query"],
    template="""You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions in JSON format (as alternative_questions).
    Original question: {query}"""
)

def expand_query(query):
    response = llm.invoke(query_expansion_prompt.format(query=query), response_format={"type": "json_object"})
    expanded_queries = json.loads(response.content)
    return expanded_queries['alternative_questions']
