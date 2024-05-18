import json
import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Set API keys from environment variables
os.environ['PINECONE_API_KEY'] = '57208fe4-cd6b-45a2-83fd-12ee06690b67'
os.environ['OPENAI_API_KEY'] = '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'
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')
import os

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("kyle-ai-index")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=openai_api_key
)

# Initialize Pinecone Vector Store
text_field = "text"
vectorstore = PineconeVectorStore(index, embeddings, text_field)


def semantic_search(event, context):

    query_text = event["queryStringParameters"]["query"]

    search_results = vectorstore.search(query=query_text, k=10, search_type="similarity")
    print(search_results)

    results = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in search_results
    ]


    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  # Ensure CORS headers are present
        },
        "results": results,
    }