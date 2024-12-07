import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_app.routers import recommend

# Create FastAPI app instance
app = FastAPI(
    title="Jester Recommender API",
    description="API for Jester Recommender System with Baseline, Content-Based, Collaborative, and Session-Based recommendations.",
    version="2.0.0",
)

# Enable CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(recommend.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Jester Recommender API! Use /help to see available endpoints."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)