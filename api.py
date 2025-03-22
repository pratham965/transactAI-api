from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI(title="TransactAI")

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def hello():
    return {"server": "active"}

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.getenv("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)
