from pinecone import Pinecone
import os
from flask import jsonify, request
from dotenv import load_dotenv

load_dotenv()

# Set API keys from environment variables
pinecone_api_key = os.getenv('PINECONE_API_KEY')

def get_doc_names(database):
    """
    Fetches document names from a Pinecone index.

    Args:
        event: The event data passed to the function.
        context: The context object passed to the function.

    Returns:
        A list of document filenames.

    Raises:
        PineconeError: If there is an error fetching the titles.
    """

    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(database)

    all_filenames = set()  # For storing all unique filenames

    page_size = 2000  # Adjust as needed based on Pinecone's limit


    dummy_vector = [0.0] * 1536  # Dummy vector for search
    length = 900
    result = index.query(
            include_metadata=True,
            filter={'filename': {'$exists': True}},
            vector=dummy_vector,
            top_k=length,
        )

    for item in result.matches:
        all_filenames.add(item.metadata['filename'])

    print(all_filenames)
    return jsonify(list(all_filenames))