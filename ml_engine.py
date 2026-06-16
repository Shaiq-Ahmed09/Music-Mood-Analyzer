import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

DATA_PATH = "data/278k_labelled_uri.csv"

print("Loading Machine Learning dataset into memory")
for i in range(0, 7):
    print(".")
    time.sleep(0.5)

try:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], errors = 'ignore')
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    print("Dataset loaded and features normalized successfully! Engine READY!")

except Exception as e:
    print(f"CRITICAL ERROR: Could not load the dataset. Make sure it is in {DATA_PATH}. Error: {e}")
    df = None

MOOD_MAPPING = {
    "sad": 0,
    "happy": 1,
    "energetic": 2,
    "calm": 3
}

def get_recommendations (user_mood: str, limit: int = 5):
    """Finds the closest matching songs based on the users mood."""

    if df is None:
        return {"error": "Dataset is not loaded . Check server logs."}
    
    mood_label = MOOD_MAPPING.get(user_mood.lower())
    if mood_label is None:
        return {"error": f"Mood '{user_mood}' is not recognized. Try: sad, happy, energetic, calm."}
    
    mood_subset = df[df['labels'] == mood_label].reset_index(drop = True)

    if mood_label == 1: #happy
        ideal_vector = np.array([[0.8, 0.8, 0.8, 0.2, 0.1, 0.0, 0.2, 0.9, 0.6]])
    elif mood_label == 0: #sad
        ideal_vector = np.array([[0.3, 0.2, 0.3, 0.1, 0.8, 0.1, 0.1, 0.1, 0.3]])
    elif mood_label == 2: #Energetic
        ideal_vector = np.array([[0.7, 0.9, 0.9, 0.3, 0.0, 0.1, 0.4, 0.6, 0.8]])
    elif mood_label == 3: #calm
        ideal_vector = np.array([[0.4, 0.2, 0.2, 0.1, 0.9, 0.5, 0.1, 0.5, 0.2]])

    subset_features = mood_subset[features].values
    similarities = cosine_similarity(ideal_vector, subset_features).flatten()
    
    top_indices = similarities.argsort()[-limit:][::-1]
    top_tracks = mood_subset.iloc[top_indices]

    track_list = top_tracks[['uri', 'danceability', 'energy', 'valence']].to_dict(orient = "records")
    
    return {
        "mood detected": user_mood.lower(),
        "tracks": track_list
    }
