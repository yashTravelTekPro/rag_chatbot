from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# load environment variables first thing
load_dotenv()

# import routes after env is loaded
from app.api.routes import router

app = FastAPI(
    title="EzeeChatBot API",
    description="RAG-based chatbot API that answers questions from your knowledge base",
    version="1.0.0"
)

# allow CORS for testing from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# register all the routes
app.include_router(router)

@app.get("/")
def root():
    # simple health check endpoint
    return {"message": "EzeeChatBot API is running!", "status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # run the server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
