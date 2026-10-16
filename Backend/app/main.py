from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Backend.app.api.upload import router as upload_router

app = FastAPI(
    title="OmniBrain API",
    description="Backend API for document upload and processing.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register the PDF upload API
app.include_router(upload_router)

@app.get("/")
def home():
    return{
        "message":"Welcome to OmniBrain API",
        "status":"running"
    }

@app.get("/health")
def health_check():
    return {"status":"healthy"}