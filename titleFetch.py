from pinecone import Pinecone
import os
from flask import jsonify, request

# Set API keys from environment variables
os.environ['PINECONE_API_KEY'] = '57208fe4-cd6b-45a2-83fd-12ee06690b67'
pinecone_api_key = os.environ.get('PINECONE_API_KEY')

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