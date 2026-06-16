from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ml_engine import get_recommendations

app = FastAPI(
    title= "Music Mood Analyzer API",
    description= "An API that recommends Spotify tracks based on human moods."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.get("/")
def read_root():
    return {"status": "Online", "message": "Welcome to the Music Mood Analyzer backend"}

@app.get("/api/recommend")
def recommend_endpoint(mood: str, limit: int = 5):
    results = get_recommendations(mood, limit)
    return results

if __name__ == "__main__":
    uvicorn.run("app:app", host = "127.0.0.1", port = 8000, reload = True)
