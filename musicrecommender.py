import streamlit as st
import pandas as pd
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# --- SET PAGE CONFIG FIRST ---
st.set_page_config(page_title="🎵 Music Recommender", layout="wide")

# --- SESSION STATE SETUP ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# --- LOAD USER DATA ---
USERS_FILE = "users.csv"
if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["username", "email", "password"]).to_csv(USERS_FILE, index=False)
users_df = pd.read_csv(USERS_FILE)

# --- STYLING ---
st.markdown(
    """
    <style>
    .stTextInput > div > input {
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- APP HEADER ---
st.markdown("<h2 style='text-align:center;'>🎵 Music Track Recommendation System</h2>", unsafe_allow_html=True)

# --- AUTHENTICATION ---
if not st.session_state.logged_in:
    auth_mode = st.radio("Select Option", ["Login", "Sign Up"], horizontal=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if auth_mode == "Login":
            st.subheader("🔐 Login")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login"):
                if not email or not password:
                    st.error("Please fill in all fields.")
                else:
                    match = users_df[
                        (users_df["email"].str.strip() == email.strip()) &
                        (users_df["password"].str.strip() == password.strip())
                    ]
                    if match.empty:
                        st.error("Account not present. Please sign up first.")
                    else:
                        st.session_state.logged_in = True
                        st.session_state.user_email = email.strip()
                        st.rerun()

        else:
            st.subheader("📝 Sign Up")
            username = st.text_input("Username", key="signup_username")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")

            if st.button("Sign Up"):
                if not username or not email or not password:
                    st.error("Please fill in all fields.")
                elif email.strip() in users_df["email"].str.strip().values:
                    st.error("Email already registered. Please log in.")
                else:
                    new_user = pd.DataFrame(
                        [[username.strip(), email.strip(), password.strip()]],
                        columns=["username", "email", "password"]
                    )
                    new_user.to_csv(USERS_FILE, mode="a", index=False, header=False)
                    st.success("Account created! You can now log in.")
                    time.sleep(2)  # 👈 Show message for 2 seconds before rerun
                    st.rerun()

# --- MAIN APP ---
if st.session_state.logged_in:
    @st.cache_data
    def load_data():
        return pd.read_csv("SpotifyFeatures.csv")

    @st.cache_data
    def prepare_model(df):
        features = df.select_dtypes(include=['float64', 'int64']).drop(['duration_ms'], axis=1)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        model = NearestNeighbors(n_neighbors=6, metric='cosine')
        model.fit(features_scaled)
        return model, features_scaled

    df = load_data()
    model, features_scaled = prepare_model(df)

    tab1, tab2 = st.tabs(["🔍 Recommend", "📊 Track Info"])

    with tab1:
        with st.sidebar:
            artist_names = sorted(df['artist_name'].unique())
            selected_artist = st.selectbox("🎤 Choose Artist", artist_names)

            filtered_tracks = df[df['artist_name'] == selected_artist]['track_name'].unique()
            selected_track = st.selectbox("🎼 Choose Track", sorted(filtered_tracks))

        song_index = df[(df['artist_name'] == selected_artist) & (df['track_name'] == selected_track)].index[0]
        distances, indices = model.kneighbors([features_scaled[song_index]])

        st.subheader(f"🎧 Recommended Tracks similar to '{selected_track}' by {selected_artist}")
        for i in range(1, len(indices[0])):
            index = indices[0][i]
            song = df.iloc[index]

            col1, col2 = st.columns([1, 4])
            with col1:
                if 'image_url' in song:
                    st.image(song['image_url'], width=100)
            with col2:
                st.markdown(f"**{song['track_name']}** by *{song['artist_name']}*")
                if 'preview_url' in song and pd.notna(song['preview_url']):
                    st.audio(song['preview_url'])

        if st.button("🎯 This was a good recommendation!"):
            st.success("Thanks for your feedback!")

    with tab2:
        st.subheader("📊 Selected Track Features")
        features_to_show = ['danceability', 'energy', 'tempo', 'valence']
        song = df.iloc[song_index]

        for f in features_to_show:
            st.metric(label=f.capitalize(), value=round(song[f], 3))

    if st.button("🔓 Logout"):
        st.session_state.logged_in = False
        st.rerun()
