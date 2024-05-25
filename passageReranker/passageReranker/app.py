import json
import os
import concurrent.futures
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI  # Correct import for ChatOpenAI

pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

rerank_template = PromptTemplate(
    input_variables=["question", "passage"],
    template="""You are an AI document storing bot. Your job is to take in a question and a passage from a document, and return its relevance score from 0 to 100, with 100 being extremely relevant (containing the answer to the question) and 0 being extremely irrelevant.
    QUESTION: {question}
    PASSAGE: {passage}

    Return the score as 'score' in json format (for example: 'score': 85)
    """
)

def rerank_passage(llm, query, passage):
    # Create the input for the LLMChain
    prompt_input = {
        "question": query,
        "passage": passage,
    }

    # Create the LLMChain
    chain = LLMChain(llm=llm, prompt=rerank_template, output_key="rerank")
    response = chain.invoke(prompt_input)
    passage['score'] = response['rerank']
    return passage

def rerank_passages(query, passages, k):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, response_format={"type": "json_object"})
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_passage = {executor.submit(rerank_passage, llm, query, passage): passage for passage in passages}
        scored_passages = []
        for future in concurrent.futures.as_completed(future_to_passage):
            try:
                scored_passages.append(future.result())
            except Exception as exc:
                print(f"Generated an exception: {exc}")
    
    # Sort the passages by score in descending order and select the top 'k'
    top_k_passages = sorted(scored_passages, key=lambda x: x['score'], reverse=True)[:k]
    return top_k_passages

def lambda_handler(event, context):
    query = event['query']
    passages = event['passages']
    k = event.get('k', 20)  # Default to 20 if not provided

    top_k_passages = rerank_passages(query, passages, k)
    return {
        'statusCode': 200,
        'body': json.dumps(top_k_passages)
    }
