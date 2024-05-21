from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI  # Correct import for ChatOpenAI
import json
import os

pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

rerank_template = PromptTemplate(
    input_variables=["question", "passages", "k"],
    template="""You are an AI language model assistant. Your task is to rerank passages related to a query based on their relevance. The most relevant passages should be put at the beginning. You should only pick at max {k} passages.
    The following are passages related to this query: {question}.
    Passages: {passages}
    
    Return the numbers of the passages (the order in which they appear, the first passage would be '1', the second '2', etc.) in reranked order of importance in json format (with 'order' being the key, and the value being the list of the passages. An example would be 'order':[1, 5, 7, 9, 11])."""
)

def rerank_passages(query: str, passages: list[str], k: int) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=openai_api_key, response_format={ "type": "json_object" })  # Correct model and API key
    formatted_passages = "\n".join(passages)  # Join passages into a single string

    # Create the input for the LLMChain
    prompt_input = {
        "question": query,
        "passages": formatted_passages,
        "k": k
    }

    # Create the LLMChain
    chain = LLMChain(llm=llm, prompt=rerank_template, output_key="rerank")
    response = chain.invoke(prompt_input)
    response_json = json.loads(response['rerank'])
    # print(response_json)
    
    return response_json
