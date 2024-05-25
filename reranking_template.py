from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI  # Correct import for ChatOpenAI
import json
import os

pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

rerank_template = PromptTemplate(
    input_variables=["question", "passage", "k"],
    template="""You are an AI document storing bot. Your job is to tak in a question and a passage from a document, and return its relevance score from 0 to 100, with 100 being extremely relevant (containing the answer to the question) and 0 being extremely irrelevant.
    QUESTION: {question}
    PASSAGE: {passage}

    Return the score as 'score' in json format (for example: 'score': score)
    """)

def rerank_passages(query: str, passages: list[dict], k: int):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, response_format={ "type": "json_object" })  # Correct model and API key

    for passage in passages: 

        # Create the input for the LLMChain
        prompt_input = {
            "question": query,
            "passage": passage,
            "k": k
        }

        # Create the LLMChain
        chain = LLMChain(llm=llm, prompt=rerank_template, output_key="rerank")
        response = chain.invoke(prompt_input)
        passage['score'] = response['rerank']

    top_k_passages = [item for item in passages] ## PULL TOP K PASSAGES, CODE NOT IMPLEMENTED YET
    # print(response_json)
    
    return top_k_passages
