import json
from langchain_openai import OpenAIEmbeddings
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone
import os
import nltk

nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

bm25 = BM25Encoder.default()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key
)

index_name = 'hybrid'
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

def hybrid_score_norm(dense, sparse, alpha: float):
    """Hybrid score using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: scale between 0 and 1
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hs = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    return [v * alpha for v in dense], hs


def hybrid_search(query):
    sparse_vector = bm25.encode_queries(query)
    dense_vector = embeddings.embed_query(query)

    hdense,hsparse = hybrid_score_norm(dense_vector, sparse_vector, alpha=0.75)

    search_results = index.query(
        top_k=20,
        vector=hdense,
        sparse_vector=hsparse,
        include_metadata=True
    )

    docs = [doc['metadata'] for doc in search_results.matches]
    return docs

def lambda_handler(event, context):
    query = event['query']
    search_results=[]
    
    # Perform query expansion
    for question in query:
        search_results.extend(hybrid_search(question))
    
    return {
        'statusCode': 200,
        'body': json.dumps({'search_results': search_results})
    }