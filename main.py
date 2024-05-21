import json
import boto3
from pinecone import Pinecone
from openai import OpenAI
import os
from datetime import datetime
from flask import request, Response
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from query_expansion import expand_query
from self_query_improvement import hybrid_search
from reranking_template import rerank_passages
from langchain.callbacks import StdOutCallbackHandler
from dotenv import load_dotenv

load_dotenv('C:/Users/Dan/OneDrive/Documents/Crosslinx/AI_Web/backend/CASSIE_Backend/.env')


# Set API keys from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

print(openai_api_key)
# Define prompt templates
LLM_CHAIN_PROMPT = PromptTemplate(
    input_variables=["question", "chat_history"],
    template="""Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
## Conversation History:
{chat_history}
## Query:
{question}
Standalone question:""",
)

ANSWER_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""## CONTEXT:
{context}
# QUERY:
{question}
## INSTRUCTIONS:
You are an AI language model assistant. Use the provided context to answer the query. If the context does not provide enough information, indicate that additional information is needed.
## ANSWER:""",
)

check_prompt = PromptTemplate(
    input_variables=["question"],
    template="Does the following question require looking up external information or can it be answered directly? Please respond with 'yes' or 'no'.\n\nQuestion: {question}",
)

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="result",
    return_messages=True,
)

# Initialize AWS DynamoDB clients
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("ChatSessions")
userTable = dynamodb.Table("userSessions")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=openai_api_key
)

# Initialize ChatOpenAI
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)

answer_llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-4o",
    temperature=0,
    stream=True,  # Enable streaming
)


def create_search_kwargs(filters):
    search_kwargs = {"k": 15}
    if filters:
        search_kwargs["filter"] = {"document_title": {"$in": filters}}
    return search_kwargs


def load_messages_into_memory(messages, memory):
    for message in messages:
        memory.chat_memory.add_user_message(message["question"])
        memory.chat_memory.add_ai_message(message["response"])


def get_past_messages(sessionId):
    try:
        response = table.get_item(Key={"sessionId": sessionId})
        if "Item" in response:
            return response["Item"].get("messages", [])
        else:
            return []
    except Exception as e:
        print(f"Error fetching from DynamoDB: {str(e)}")
        raise e


def save_to_dynamodb(sessionId, timestamp, query, bot_response):
    try:
        response = table.update_item(
            Key={"sessionId": sessionId},
            UpdateExpression="SET lastUpdated = :timestamp, messages = list_append(if_not_exists(messages, :empty_list), :new_messages)",
            ExpressionAttributeValues={
                ":timestamp": timestamp,
                ":new_messages": [
                    {
                        "timestamp": timestamp,
                        "question": query,
                        "response": bot_response,
                    }
                ],
                ":empty_list": [],
            },
            ReturnValues="UPDATED_NEW",
        )
    except Exception as e:
        print(f"Error updating DynamoDB: {str(e)}")
        raise e


def save_to_user_dynamodb(sessionId, timestamp, userId):
    try:
        response = userTable.get_item(Key={"userId": userId})
        user_item = response.get("Item")

        expression_attribute_names = {}
        expression_attribute_values = {}

        if user_item:
            sessions = user_item.get("sessions", [])
            session_exists = next(
                (session for session in sessions if session["sessionId"] == sessionId),
                None,
            )

            if session_exists:
                session_index = sessions.index(session_exists)
                update_expression = f"SET sessions[{session_index}].#ts = :timestamp"
                expression_attribute_names = {"#ts": "timestamp"}
                expression_attribute_values = {":timestamp": timestamp}
            else:
                update_expression = (
                    "SET sessions = list_append(sessions, :new_sessions)"
                )
                expression_attribute_values = {
                    ":new_sessions": [{"sessionId": sessionId, "timestamp": timestamp}]
                }
        else:
            update_expression = "SET sessions = if_not_exists(sessions, :empty_list)"
            expression_attribute_values = {
                ":empty_list": [{"sessionId": sessionId, "timestamp": timestamp}]
            }

        update_params = {
            "Key": {"userId": userId},
            "UpdateExpression": update_expression,
            "ExpressionAttributeValues": expression_attribute_values,
            "ReturnValues": "UPDATED_NEW",
        }
        if expression_attribute_names:
            update_params["ExpressionAttributeNames"] = expression_attribute_names

        response = userTable.update_item(**update_params)
        print(f"UpdateItem succeeded: {json.dumps(response, indent=4)}")
    except Exception as e:
        print(f"Error updating DynamoDB: {str(e)}")
        raise e


def handle_question(question):
    expanded_queries = expand_query(question)
    all_results = []
    results_with_metadata = []
    results_with_metadata_reranked = []

    # Using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(hybrid_search, q): q for q in expanded_queries
        }
        for future in as_completed(future_to_query):
            query = future_to_query[future]
            try:
                hybrid_results = future.result()
                all_results.extend([result.page_content for result in hybrid_results])
                results_with_metadata.extend(hybrid_results)
            except Exception as exc:
                print(f"Hybrid search generated an exception: {exc}")

    reranked_passages = rerank_passages(question, all_results, k=5)
    print(reranked_passages)
    reranked_results = []
    context_string = ""
    print(results_with_metadata)
    for i in reranked_passages["order"]:
        print(all_results[i - 1])
        reranked_results.append(all_results[i - 1])
        results_with_metadata_reranked.append(results_with_metadata[i - 1])
        for i in results_with_metadata_reranked:
            context_string += i.page_content + "\n"


    answer_chain = LLMChain(llm=answer_llm, prompt=ANSWER_PROMPT, output_key="answer")

    def generate():
        for token in answer_llm.stream(
            f"""## CONTEXT:
            {context_string}\n
            # QUERY:
            {question}
            ## INSTRUCTIONS:
            You are an AI language model assistant. Use the provided context to answer the query. If the context does not provide enough information, indicate that additional information is needed.
            ## ANSWER:"""
        ):
            print(token)
            if '\n' in token.content:
                token.content = token.content.replace('\n', '<br>')
            print(token.content)
            yield f"data: {token.content}\n\n"
        
        yield 'event: end\ndata: end\n\n'

    source_documents = [
        {"source": {"page_content": result.page_content, "metadata": result.metadata}}
        for result in results_with_metadata_reranked
    ]

    return generate, source_documents


def send_message(event, context):
    memory.clear()

    # Extract parameters from the request
    sessionId = request.args.get("sessionId")
    query = request.args.get("query")
    userId = request.args.get("userId")
    database = request.args.get("database")
    filters = request.args.getlist("filters")

    # Load past messages into memory
    past_messages = get_past_messages(sessionId)
    load_messages_into_memory(past_messages, memory)

    # Rephrase the question
    question_chain = LLMChain(llm=llm, prompt=LLM_CHAIN_PROMPT, output_key="query")
    rephrased_question = question_chain(
        {"question": query, "chat_history": memory.buffer_as_str}
    )
    print(rephrased_question["query"])

    # Generate a response using the RetrievalQA chain
    generate, source_documents = handle_question(rephrased_question["query"])

    # Get current timestamp
    timestamp = datetime.utcnow().isoformat()

    # Save the query and response to DynamoDB
    save_to_dynamodb(sessionId, timestamp, query, "")
    save_to_user_dynamodb(sessionId, timestamp, userId)

    # Return the response
    return Response(generate(), mimetype="text/event-stream")
