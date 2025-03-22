import fastapi
import uvicorn

app = fastapi.FastAPI(title="TransactAI")

app.add_middleware(
    fastapi.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def hello():
    return {"server":"active"}
