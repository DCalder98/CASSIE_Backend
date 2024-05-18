from flask import Flask, request, jsonify
from conversational_ai import send_message
from semantic_search import semantic_search
from titleFetch import get_doc_names
from docUpload import doc_upload
from flask_cors import CORS, cross_origin

app = Flask(__name__)

# Example Route
@app.route('/', methods=['GET'])
def test():
    return jsonify({'message': 'Hello, World!'})


@app.route('/query', methods=['GET'])
@cross_origin()
def handle_query():
    sessionId = request.args.get('sessionId')
    query = request.args.get('query')
    userId = request.args.get('userId')

    event = {
        "queryStringParameters": {
            "sessionId": sessionId,
            "query": query,
            "userId": userId
        }
    }
    context = None

    response = send_message(event, context)
    return response


@app.route('/sem_search', methods=['GET'])
@cross_origin()
def handle_search():
    query = request.args.get('query')

    event = {
        "queryStringParameters": {
            "query": query,
        }
    }
    context = None

    response = semantic_search(event, context)
    return jsonify(response)


@app.route('/get_doc_names', methods=['GET'])
@cross_origin()
def handle_get_doc_names():
    database = request.args.get('database', 'new-cx-ai')
    try:
        print('Getting doc names')
        document_titles = get_doc_names(database)
        return document_titles
    except Exception as e:
        return jsonify(error=str(e)), 500
    
@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        doc_upload()
        return jsonify({'message': 'File uploaded successfully!'})


if __name__ == '__main__':
    app.run(debug=True)

CORS(app) 