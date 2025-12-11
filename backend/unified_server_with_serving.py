from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException
from starlette.responses import RedirectResponse
import os
import uvicorn
from pathlib import Path

# Import the API routers from the existing app
from app.routers import query, auth, personalize, translate, quizzes

# Create the main FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics Educational Platform API",
    description="Unified server for the educational platform with RAG chatbot, authentication, personalization, and translation features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose headers for client-side access
    expose_headers=["Access-Control-Allow-Origin"]
)

# Include the existing API routers
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(personalize.router, prefix="/api", tags=["personalize"])
app.include_router(translate.router, prefix="/api", tags=["translate"])
app.include_router(quizzes.router, prefix="/api", tags=["quizzes"])

# Define the frontend build directory
frontend_build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "build")

# Mount static files if the build directory exists
if os.path.exists(frontend_build_dir):
    app.mount("/static", StaticFiles(directory=os.path.join(frontend_build_dir, "static")), name="static")
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_build_dir, "assets")), name="assets")
else:
    print("Warning: Frontend build directory does not exist. Run 'npm run build' in the frontend directory to build the frontend.")

@app.get("/")
async def root():
    if os.path.exists(frontend_build_dir):
        return FileResponse(os.path.join(frontend_build_dir, "index.html"))
    else:
        return {"message": "Physical AI & Humanoid Robotics Educational Platform API - Frontend not built yet"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Physical AI & Humanoid Robotics API"}

# Fallback to return index.html for client-side routing (excluding API routes)
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # If the path starts with /api/, let the API routers handle it through the existing routes
    if full_path.startswith("api/"):
        # This should be handled by the API routes already defined
        # Return a redirect to trigger the actual API route
        return RedirectResponse(url=f"/{full_path}")

    # For all other paths, serve the frontend
    if os.path.exists(frontend_build_dir):
        # Check if it's a static file (like assets, css, js)
        file_path = os.path.join(frontend_build_dir, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)

        # For client-side routing (docs, tutorials, etc.), return index.html
        return FileResponse(os.path.join(frontend_build_dir, "index.html"))
    else:
        return {"message": "Frontend not built yet. Run 'npm run build' in the frontend directory."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)