import json
import boto3
import os
from datetime import datetime
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from flask import request

# Set API keys from environment variables
os.environ['PINECONE_API_KEY'] = '57208fe4-cd6b-45a2-83fd-12ee06690b67'
os.environ['OPENAI_API_KEY'] = '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'
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Define prompt templates
LLM_CHAIN_PROMPT = PromptTemplate(
    input_variables=["question", 'chat_history'],
    template="""Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
## Conversation History:
{chat_history}
## Query:
{question}
Standalone question:"""
)

RETRIEVAL_PROMPT = PromptTemplate(
    input_variables=["question", 'context'],
    template="""## CONTEXT:
{context}
# QUERY:
{question}
## INSTRUCTIONS:
You are a retrieval chatbot. Your main task is to answer the above query using the context provided. The context is from 10 chunks, taken from a repository of documents and tables. If you do not know the answer, you can say "I don't know".
## ANSWER:"""
)

REVIEW_PROMPT = PromptTemplate(
    input_variables=["question", 'answer'],
    template="""## Answer:
{answer}
# QUERY:
{question}
## INSTRUCTIONS:
You are a third-party reviewer who serves as a consultant for a large business. A department has answered a question using the knowledge in their database. Your job is to review the answer and check that it is sufficient.
## IMPORTANT:
If the answer is correct and fully answers the question, return the original answer, with no added information. Do not add things like: 'the original answer provided is correct', just return the answer verbatim. If the answer is incorrect or incomplete, use your reasoning and general knowledge to provide a corrected or complete answer.
## ANSWER:"""
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
userTable = dynamodb.Table('userSessions')

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key
)

# Initialize ChatOpenAI
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-4o",
    temperature=0
)

def create_search_kwargs(filters):
    """
    Create search keyword arguments for the retriever.
    """
    search_kwargs = {'k': 15}
    if filters:
        search_kwargs['filter'] = {'filename': {'$in': filters}}
    return search_kwargs

def load_messages_into_memory(messages, memory):
    """
    Load past messages into conversation memory.
    """
    for message in messages:
        memory.chat_memory.add_user_message(message['question'])
        memory.chat_memory.add_ai_message(message['response'])

def get_past_messages(sessionId):
    """
    Retrieve past messages for the session from DynamoDB.
    """
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
    """
    Append the query and response to the chat session in DynamoDB.
    """
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
    """
    Append the session to the user's entry in DynamoDB.
    """
    try:
        response = userTable.get_item(Key={"userId": userId})
        user_item = response.get('Item')

        expression_attribute_names = {}
        expression_attribute_values = {}

        if user_item:
            sessions = user_item.get('sessions', [])
            session_exists = next((session for session in sessions if session['sessionId'] == sessionId), None)
            
            if session_exists:
                session_index = sessions.index(session_exists)
                update_expression = f"SET sessions[{session_index}].#ts = :timestamp"
                expression_attribute_names = {'#ts': 'timestamp'}
                expression_attribute_values = {
                    ":timestamp": timestamp
                }
            else:
                update_expression = "SET sessions = list_append(sessions, :new_sessions)"
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

def send_message(event, context):
    """
    Handle incoming messages, process them using the retrieval QA pipeline, and return the response.
    """
    memory.clear()

    # Extract parameters from the request
    sessionId = request.args.get("sessionId")
    query = request.args.get("query")
    userId = request.args.get("userId")
    database = request.args.get("database")
    filters = request.args.getlist("filters")

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(database)

    # Initialize Pinecone Vector Store
    text_field = "raw_content"
    vectorstore = PineconeVectorStore(index, embeddings, text_field)
    search_kwargs = create_search_kwargs(filters)

    # Setup RetrievalQA
    qa = RetrievalQA.from_chain_type(
        chain_type="stuff",
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs=search_kwargs),
        return_source_documents=True,
        chain_type_kwargs={"prompt": RETRIEVAL_PROMPT},
        output_key="answer",
    )

    # Setup chains
    question_chain = LLMChain(llm=llm, prompt=LLM_CHAIN_PROMPT, output_key="query")
    review_chain = LLMChain(llm=llm, prompt=REVIEW_PROMPT, output_key="result")

    # Load past messages into memory
    past_messages = get_past_messages(sessionId)
    load_messages_into_memory(past_messages, memory)

    # Rephrase the question
    rephrased_question = question_chain({'question': query, 'chat_history': memory.buffer_as_str})

    # Generate a response using the RetrievalQA chain
    response = qa.invoke(rephrased_question['query'])

    # Extract source documents
    source_documents = [doc.metadata for doc in response['source_documents']]

    # Get current timestamp
    timestamp = datetime.utcnow().isoformat()

    # Save the query and response to DynamoDB
    save_to_dynamodb(sessionId, timestamp, query, response["answer"])
    save_to_user_dynamodb(sessionId, timestamp, userId)

    # Return the response
    return {
        "result": response["answer"],
        "sources": source_documents
    }
