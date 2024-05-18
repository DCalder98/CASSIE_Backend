import pytest
import json
from app import app  # Import your Flask app
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Test the home route."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Hello, World!' in response.data

def test_query_route(client):
    """Test the query route."""
    response = client.get('/query?sessionId=FLASK_TEST_SESSION&query=hello&userId=db93081f-7af2-4fcf-b4fe-41215e2cf5a7')
    assert response.status_code == 200
    print(response.data)

def test_search_route(client):
    """Test the query route."""
    response = client.get('/sem_search?query=hello')
    assert response.status_code == 200
    print(response.data)