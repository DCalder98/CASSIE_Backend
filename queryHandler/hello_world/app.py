import json
import boto3
from datetime import datetime

def lambda_handler(event, context):
    query = event['query']
    sessionId = event['sessionId']
    userId = event['userId']
    
    timestamp = datetime.utcnow().isoformat()
    
    # Save initial query to DynamoDB
    save_to_dynamodb(sessionId, timestamp, query, "")
    save_to_user_dynamodb(sessionId, timestamp, userId)
    
    # Invoke the Query Expansion Lambda
    lambda_client = boto3.client('lambda')
    response = lambda_client.invoke(
        FunctionName='queryExpansion',
        InvocationType='RequestResponse',
        Payload=json.dumps({'query': query})
    )
    
    expanded_queries = json.loads(response['Payload'].read())

    retrieval_response = lambda_client.invoke(
        FunctionName='hybridRetrieval',
        InvocationType='RequestResponse',
        Payload=json.dumps({'questions': expanded_queries})
    )
    
    retrieved_resources = json.loads(retrieval_response['Payload'].read())

    rerank_response = lambda_client.invoke(
        FunctionName='passageReranker',
        InvocationType='RequestResponse',
        Payload=json.dumps({'passages': retrieved_resources})
    )

    reranked_passages = json.loads(rerank_response['Payload'].read())

    answer_question = lambda_client.invoke(
        FunctionName='answerQuestion',
        InvocationType='RequestResponse',
        Payload=json.dumps({'context': reranked_passages, 'question':query})
    )

    question_answer = json.loads(answer_question['Payload'].read())
    return {
        'statusCode': 200,
        'body': json.dumps(question_answer, reranked_passages)
    }

def save_to_dynamodb(sessionId, timestamp, query, bot_response):
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table("ChatSessions")
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
    dynamodb = boto3.resource("dynamodb")
    userTable = dynamodb.Table("userSessions")
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
