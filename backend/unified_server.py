from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException
from pathlib import Path
import os
import uvicorn
import mimetypes

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

# Handle API root
@app.get("/api")
async def api_root():
    return {"message": "Physical AI & Humanoid Robotics Educational Platform API"}

# Catch-all route to serve frontend for client-side routing
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # If the path starts with /api/, let the API routers handle it
    if full_path.startswith("api/"):
        # Return a response that will trigger the appropriate API route
        return JSONResponse(content={"error": "API route should be handled by specific API endpoints"}, status_code=404)

    # Check if it's a static file (css, js, images, etc.)
    if '.' in full_path:
        file_extension = os.path.splitext(full_path)[1]
        if file_extension.lower() in ['.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2', '.ttf', '.eot']:
            file_path = os.path.join(frontend_build_dir, full_path)
            if os.path.exists(file_path):
                # Determine content type
                content_type, _ = mimetypes.guess_type(file_path)
                if content_type:
                    return FileResponse(file_path, media_type=content_type)
                else:
                    return FileResponse(file_path)

    # For all other routes, serve the index.html to enable client-side routing
    if os.path.exists(frontend_build_dir):
        return FileResponse(os.path.join(frontend_build_dir, "index.html"))
    else:
        return {"message": "Frontend not built yet. Run 'npm run build' in the frontend directory."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)