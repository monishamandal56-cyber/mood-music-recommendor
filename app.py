import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from urllib.parse import quote_plus


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset.csv")

# Remove unnecessary column if it exists
df = df.drop(
    columns=["Unnamed: 0"],
    errors="ignore"
)

# Remove rows with missing important information
df = df.dropna(
    subset=[
        "artists",
        "album_name",
        "track_name"
    ]
)


# ==========================================
# 2. FEATURES USED FOR RECOMMENDATION
# ==========================================

features = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "tempo"
]


# ==========================================
# 3. SCALE SONG FEATURES
# ==========================================

scaler = MinMaxScaler()

song_features_scaled = scaler.fit_transform(
    df[features]
)

scaled_df = pd.DataFrame(
    song_features_scaled,
    columns=features,
    index=df.index
)


# ==========================================
# 4. MOOD PROFILES
# ==========================================

mood_profiles = {

    "Happy": {
        "vector": [0.75, 0.75, 0.80, 0.30, 0.60],
        "message": "Did you win a lottery? 👀",
        "emoji": "😊"
    },

    "Sad": {
        "vector": [0.35, 0.25, 0.20, 0.70, 0.35],
        "message": "Did he/she break your heart? 💔",
        "emoji": "💔"
    },

    "Romantic": {
        "vector": [0.55, 0.45, 0.65, 0.60, 0.40],
        "message": "Ooooo… someone is in so much love! 👀💕",
        "emoji": "💕"
    },

    "Relaxed": {
        "vector": [0.40, 0.30, 0.55, 0.75, 0.35],
        "message": "Ahhh… finally taking a break from the chaos? 😌",
        "emoji": "😌"
    },

    "Angry": {
        "vector": [0.55, 0.90, 0.20, 0.15, 0.75],
        "message": "Okay… who made you angry? 👀",
        "emoji": "😤"
    },

    "Excited": {
        "vector": [0.80, 0.90, 0.75, 0.15, 0.80],
        "message": "Okayyy, what’s got you this excited? 👀",
        "emoji": "🤩"
    },

    "Sleepy": {
        "vector": [0.25, 0.15, 0.25, 0.85, 0.20],
        "message": "Are we listening to music or falling asleep to it? 😴",
        "emoji": "😴"
    }
}


# ==========================================
# 5. STREAMLIT APP
# ==========================================

st.title("🎵 Mood-Based Music Recommender")

st.write("How are you feeling today?")


# ==========================================
# 6. MOOD BUTTONS
# ==========================================

selected_mood = None

for mood in mood_profiles:

    if st.button(
        mood_profiles[mood]["emoji"] + " " + mood,
        use_container_width=True
    ):
        selected_mood = mood


# ==========================================
# 7. RECOMMENDATIONS
# ==========================================

if selected_mood is not None:

    mood_data = mood_profiles[selected_mood]

    st.subheader(
        mood_data["emoji"] + " " + selected_mood
    )

    st.write(
        mood_data["message"]
    )


    # --------------------------------------
    # Calculate distance
    # --------------------------------------

    distance = pairwise_distances(
        scaled_df,
        [mood_data["vector"]],
        metric="euclidean"
    ).flatten()


    # --------------------------------------
    # Create recommendation dataframe
    # --------------------------------------

    recommendations = df.copy()

    recommendations["distance"] = distance


    # Remove duplicate songs
    recommendations = recommendations.drop_duplicates(
        subset=[
            "track_name",
            "artists"
        ]
    )


    # Sort by closest mood
    recommendations = recommendations.sort_values(
        by="distance",
        ascending=True
    )


    # --------------------------------------
    # Select top 10
    # --------------------------------------

    top_songs = recommendations.head(10)


    st.subheader("🎵 Your Recommendations")


    # ======================================
    # 8. DISPLAY SONGS + YOUTUBE SEARCH
    # ======================================

    for i, row in enumerate(
        top_songs.itertuples(),
        start=1
    ):

        st.write(
            f"{i}. **{row.track_name}** — {row.artists}"
        )


        # Create YouTube search link
        youtube_search = (
            "https://www.youtube.com/results?search_query="
            + quote_plus(
                f"{row.track_name} {row.artists}"
            )
        )


        st.markdown(
            f"[▶️ Search on YouTube]({youtube_search})"
        )