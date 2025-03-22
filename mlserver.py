from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn
from pipeline import preprocess_data
from inference import predict

app = FastAPI(title="TransactAI")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/mlpredict")
async def ml_predict(api_data: dict = Body(...)):
    # Convert incoming JSON data to DataFrame
    df = pd.DataFrame([api_data])
    
    # Preprocess the data
    df_new = preprocess_data(df)
    
    # Make predictions
    output = predict(df_new)
    result = int(output.numpy()[0])  # Ensure it's in a serializable format
    
    return {"transaction_id": api_data.get("transaction_id_anonymous", ""), "is_fraud": result}

if __name__ == "__main__":
    port = 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
