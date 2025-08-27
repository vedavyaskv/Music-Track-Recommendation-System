import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os

st.set_page_config(page_title="🎵 Music Recommender", layout="wide")

st.markdown("<h2 style='text-align:center;'>🎵 Music Track Recommendation System</h2>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("SpotifyFeatures.csv")

@st.cache_data
def prepare_model(df):
    features = df.select_dtypes(include=['float64', 'int64']).drop(['duration_ms'], axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = NearestNeighbors(n_neighbors=20, metric='cosine')
    model.fit(features_scaled)
    return model, features_scaled

def save_favourite(track_name, artist_name):
    fav_file = "favourites.csv"
    new_entry = pd.DataFrame([[track_name, artist_name]], columns=["Track", "Artist"])
    if os.path.exists(fav_file):
        fav_df = pd.read_csv(fav_file)
        if not ((fav_df["Track"] == track_name) & (fav_df["Artist"] == artist_name)).any():
            fav_df = pd.concat([fav_df, new_entry], ignore_index=True)
            fav_df.to_csv(fav_file, index=False)
    else:
        new_entry.to_csv(fav_file, index=False)

def remove_favourite(track_name, artist_name):
    fav_file = "favourites.csv"
    if os.path.exists(fav_file):
        fav_df = pd.read_csv(fav_file)
        fav_df = fav_df[~((fav_df["Track"] == track_name) & (fav_df["Artist"] == artist_name))]
        fav_df.to_csv(fav_file, index=False)

df = load_data()
model, features_scaled = prepare_model(df)

if os.path.exists("favourites.csv"):
    fav_df = pd.read_csv("favourites.csv")
    st.session_state.added_favs = set(f"{t}|{a}" for t, a in zip(fav_df["Track"], fav_df["Artist"]))
else:
    st.session_state.added_favs = set()

if "confirm_remove" not in st.session_state:
    st.session_state.confirm_remove = None

tab1, tab2, tab3 = st.tabs(["🔍 Recommend", "📊 Track Info", "❤️ Favourites"])

with tab1:
    with st.sidebar:
        st.markdown("### 🎯 Choose Recommendation Mode")
        mode = st.radio("Select Mode", ["By Artist & Track", "By Mood"])

        if mode == "By Artist & Track":
            artist_names = sorted(df['artist_name'].unique())
            selected_artist = st.selectbox("Artist", artist_names)
            filtered_tracks = df[df['artist_name'] == selected_artist]['track_name'].unique()
            selected_track = st.selectbox("Track", sorted(filtered_tracks))
        else:
            mood = st.selectbox("Select Mood", ["Happy", "Sad", "Energetic", "Calm"])
            if mood == "Happy":
                mood_df = df[(df['valence'] > 0.6) & (df['energy'] > 0.6)]
            elif mood == "Sad":
                mood_df = df[(df['valence'] < 0.4) & (df['energy'] < 0.4)]
            elif mood == "Energetic":
                mood_df = df[(df['energy'] > 0.7) & (df['valence'] > 0.4) & (df['valence'] < 0.7)]
            else:
                mood_df = df[(df['energy'] < 0.5) & (df['valence'] > 0.4) & (df['valence'] < 0.7)]
            selected_track = st.selectbox("Track", sorted(mood_df['track_name'].unique()))
            selected_artist = mood_df[mood_df['track_name'] == selected_track]['artist_name'].iloc[0]

        exclude_same_artist = st.checkbox("Exclude same artist from recommendations", value=False)

    song_index = df[(df['artist_name'] == selected_artist) & (df['track_name'] == selected_track)].index[0]
    distances, indices = model.kneighbors([features_scaled[song_index]])

    st.subheader(f"🎧 Recommended Tracks similar to '{selected_track}' by {selected_artist}")

    shown_songs = set()
    for i in range(1, len(indices[0])):
        index = indices[0][i]
        song = df.iloc[index]

        if exclude_same_artist and song['artist_name'] == selected_artist:
            continue
        if song['track_name'] in shown_songs:
            continue

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{song['track_name']}** by *{song['artist_name']}*")
            if 'preview_url' in song and pd.notna(song['preview_url']):
                st.audio(song['preview_url'])
        with col2:
            song_key = f"{song['track_name']}|{song['artist_name']}"
            if song_key in st.session_state.added_favs:
                st.button("💖 Added", key=f"added_{i}", disabled=True)
            else:
                if st.button("❤️ Add", key=f"fav_{i}"):
                    save_favourite(song['track_name'], song['artist_name'])
                    st.session_state.added_favs.add(song_key)
                    st.rerun()  # 🔹 Instant update

        shown_songs.add(song['track_name'])
        if len(shown_songs) >= 5:
            break

    if not shown_songs:
        st.warning("No unique recommendations found.")

with tab2:
    st.subheader("📊 Selected Track Features")
    features_to_show = ['danceability', 'energy', 'tempo', 'valence']
    song = df.iloc[song_index]
    cols = st.columns(len(features_to_show))
    for i, f in enumerate(features_to_show):
        cols[i].metric(label=f.capitalize(), value=round(song[f], 3))

with tab3:
    st.subheader("❤️ Your Favourite Songs")
    fav_file = "favourites.csv"
    if os.path.exists(fav_file):
        fav_df = pd.read_csv(fav_file)

        if not fav_df.empty:
            for idx, row in fav_df.iterrows():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{row['Track']}** by *{row['Artist']}*")
                with col2:
                    if st.button("🗑 Remove", key=f"del_{idx}"):
                        st.session_state.confirm_remove = (row['Track'], row['Artist'])
                        st.rerun()

            if st.session_state.confirm_remove:
                track_to_remove, artist_to_remove = st.session_state.confirm_remove
                st.warning(f"Are you sure you want to remove **{track_to_remove}** by *{artist_to_remove}*?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Yes, remove"):
                        remove_favourite(track_to_remove, artist_to_remove)
                        song_key = f"{track_to_remove}|{artist_to_remove}"
                        if song_key in st.session_state.added_favs:
                            st.session_state.added_favs.remove(song_key)
                        st.session_state.confirm_remove = None
                        st.rerun()
                with c2:
                    if st.button("❌ Cancel"):
                        st.session_state.confirm_remove = None
                        st.rerun()
        else:
            st.info("No favourites added yet. Add some from the recommendations tab!")
    else:
        st.info("No favourites added yet. Add some from the recommendations tab!")
