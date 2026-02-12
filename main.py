from fastapi import FastAPI
from pydantic import BaseModel, Field
from logic import FraudDetectionSystem

# 1. Create an instance of our logic
detector = FraudDetectionSystem()

# 2. Initialize the FastAPI application
app = FastAPI(title="Fraud Detection API")

# 3. Describe the structure of input data for validation
class Transaction(BaseModel):
    # Amount must be greater than 0
    amount: float = Field(..., gt=0, example=100.50)
    # Hour must be between 0 and 23
    transaction_hour: int = Field(..., ge=0, le=23, example=14)
    # Distance cannot be negative
    distance_from_home: float = Field(..., ge=0, example=5.0)

# 4. Create an endpoint for transaction checking
@app.post("/predict")
def predict_fraud(data: Transaction):
    # Now results is a ready list of filtered neighbors
    results = detector.predict(
        raw_data=[data.amount, data.transaction_hour, data.distance_from_home]
    )
    
    # Format the response as simply as possible
    return {
        "status": "success",
        "found_neighbors": len(results), # Just count the number of elements in the list
        "neighbors": results # Return the list, which already contains id, is_fraud, and distance
    }