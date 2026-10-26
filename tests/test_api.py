#import TestClient so we send simulated HTTP request directly to FastAPI
from fastapi.testclient import TestClient

#import the application object defined in main
from main import app

#create a resuable test client for all test functions
client = TestClient(app)

def test_health_endpoint():
    '''Verify that the application is running and can respond to a health check request'''
    response = client.get("/health")
    # A successfull http request should return a status code 200
    assert response.status_code == 200
    
    # Now check response body
    assert response.json() == {"status":"healthy"}
    
def test_valid_iris_prediction():
    '''
    Verify that valid flower measurement preduce a prediction
    '''
    payload = {
        "sepal_length": 1,
        "sepal_width": 1,
        "petal_length": 1,
        "petal_width": 1
    }
    # Send a simulated HTTP POST request with JSON input
    response = client.post("/predict",json=payload)
    #covert the json response into python dictionary
    result = response.json()
    
    # Check the response contract: both fields must exist
    assert "prediction" in result
    assert "class_name" in result
    
    # The predicted class must be one of the three iris classes
    assert result["class_name"] in ["setosa", "versicolor", "virginica"]
    
    #The predicted integer must be either 0 ,1 or 2
    assert result["prediction"] in [0,1,2]
    
def test_invalid_iris_input():
    '''
    Verify that ivalid inputs are rejected by the FastAPI
    '''
    payload = {
            "sepal_length": -1,
            "sepal_width": 1,
            "petal_length": 1,
            "petal_width": 1
        }
    # Send a simulated HTTP POST request with JSON input
    response = client.post("/predict",json=payload)
    #FastAPI normally returns HTTP 422 for validation errors
    assert response.status_code == 422
    
    
    
    
    
