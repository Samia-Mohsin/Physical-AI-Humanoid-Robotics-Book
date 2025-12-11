from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import routers
from .routers import query, auth, personalize, translate, quizzes

# Create FastAPI app instance
app = FastAPI(
    title="Physical AI & Humanoid Robotics Educational Platform API",
    description="API for the educational platform with RAG chatbot, authentication, personalization, and translation features",
    version="1.0.0"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose headers for client-side access
    expose_headers=["Access-Control-Allow-Origin"]
)

# Include routers
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(personalize.router, prefix="/api", tags=["personalize"])
app.include_router(translate.router, prefix="/api", tags=["translate"])
app.include_router(quizzes.router, prefix="/api", tags=["quizzes"])

@app.get("/")
async def root():
    return {"message": "Physical AI & Humanoid Robotics Educational Platform API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Physical AI & Humanoid Robotics API"}