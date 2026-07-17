# Music Mood Analyzer & Custom Player

A full-stack, machine-learning-powered application that analyzes human moods and recommends personalized Spotify tracks based on acoustic features (Valence, Energy, Danceability, etc.).

This repository also includes AuraStream, an entirely custom-built React music player that allows users to upload and stream local audio files in-memory with a sleek, modern UI.

# Features

ML Recommendation Engine: Uses scikit-learn (MinMaxScaler, Cosine Similarity) to map requested moods to an ideal 9-dimensional acoustic vector.

Dynamic Pool Sampling: Returns a fresh, randomized subset of the top 50 closest track matches every time a mood is requested.

FastAPI Backend: A lightning-fast, asynchronous Python backend with auto-generated Swagger UI documentation.

Spotify Integration: Renders playable Spotify <iframe/> widgets directly on the frontend using raw track URIs.

AuraStream Custom Player: A standalone React frontend demonstrating complex state management (volume, progress bars, file streaming) for local .mp3/.wav files.

# Tech Stack

Backend: Python, FastAPI, Uvicorn

Machine Learning: Pandas, Scikit-Learn, Numpy

Frontend (Analyzer): Vanilla HTML, JavaScript, Tailwind CSS (CDN)

Frontend (AuraStream): React.js, Tailwind CSS, Lucide Icons

# Getting Started (Backend ML Engine)

1. Download the Dataset

Due to GitHub size limits, the 278k Spotify Audio Features dataset is not included in this repo.

Download the 278k_labelled_uri.csv dataset from Kaggle.

Create a folder named data/ in the root directory.

Place the .csv file inside the data/ folder.

2. Set up the Environment

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

3. Run the Server

python app.py

The API will be live at http://127.0.0.1:8000. You can test the endpoints interactively via the Swagger UI at http://127.0.0.1:8000/docs.

4. Run the Frontend

Simply open index.html in your web browser or run it via a Live Server extension to see the Mood Analyzer UI.

# AuraStream (React Custom Player)

To run the custom standalone React player:

Navigate to the React app directory (if separated) or use Vite/Create-React-App to serve App.jsx.

Upload local audio files via the UI to begin streaming.

# How the Machine Learning Works

The engine uses Content-Based Filtering. When a user selects a mood (e.g., "Energetic"), the backend defines an "Ideal Target Vector" (High Energy, High Tempo, Low Acousticness). It then uses Cosine Similarity to compare this ideal vector against 278,000 tracks, instantly finding the closest matching songs based on their acoustic features.

Built as a portfolio showcase demonstrating ML pipelines, backend architecture, and modern frontend UX.
