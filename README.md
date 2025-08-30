# Music Track Recommendation System
App link: https://music-track-recommendation-system-5t3aiebg7hkniezes6s5t4.streamlit.app/

A Streamlit web app that recommends music tracks based on audio feature similarity using the Spotify Features dataset. The system leverages K-Nearest Neighbors with cosine similarity to deliver personalized suggestions, mood-based filtering, and interactive track feature insights.

## Features:

- **Personalized Recommendations** – Get similar tracks based on the selected song using KNN and cosine similarity.
- **Mood-Based Filtering** – Filter recommendations based on mood (e.g., happy, energetic, calm).
- **Audio Feature Metrics** – View key track features like energy, danceability, and tempo.
- **Favourites Management** – Add/remove songs from favourites in real-time.
- **Responsive UI** – Built with Streamlit for a fast and interactive experience.


## Tech Stack:
- Python
- Streamlit
- Pandas
- scikit-learn (Nearest Neighbors, StandardScaler)

## Dataset:
- **Source**: Spotify Features Dataset
- **Description**: Contains audio features (energy, danceability, loudness, tempo, etc.) for thousands of songs.
