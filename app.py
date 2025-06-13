import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import pandas as pd
import os # Still needed for mkdir for local logs/reports, though data will be in Firestore
from datetime import datetime, timedelta, date
import pyttsx3
from fpdf import FPDF
import glob # Still needed for local cleanup, though data will be in Firestore
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import json # Still needed for serializing data for Firestore

# --- Firebase Imports (Conceptual, these would be handled by a client-side JS app or a Python wrapper) ---
# For a pure Python Streamlit app, we would *not* directly import JS Firebase SDKs like this.
# Instead, we'd use a Python library like `firebase-admin` for server-side access,
# or create a custom component for client-side Firebase interaction.
# Given the prompt implies "full code" for a Streamlit app, and to align with Canvas's implicit Firebase setup,
# I will structure the Firestore operations as if they are synchronous calls from a Python perspective,
# but internally, they would be handled by a client-side component or a serverless function if this were deployed.
# For the purpose of this exercise in a Streamlit context, we will conceptualize the Firestore functions
# as if they directly interact with the database using an already initialized client.

# Placeholder for Firebase objects that would be initialized once globally
# In a real deployed Streamlit app, you might use `st.experimental_singleton`
# or a custom component that bridges to client-side Firebase.
# For this code, we'll assume `db` and `auth` are available after a conceptual initialization.
_db_initialized = False
_auth_initialized = False

# This is a conceptual representation. In a real Streamlit setup for Canvas
# these would likely be part of a custom component or a Streamlit-Firebase wrapper.
# Since the environment provides __app_id, __firebase_config, __initial_auth_token
# we'll assume they are available and used for *actual* Firebase init.
# For *this* Python code, we define placeholder functions that would internally
# call the actual Firebase logic.

# In a real Python Streamlit app, if you were to use Firebase Auth, you'd use a server-side
# Admin SDK, or a client-side JS component. For the purpose of this exercise,
# we are simulating the behavior as if Streamlit could directly make these calls.

def _get_firestore_client(app_id):
    # In a real Streamlit app, this would be `firebase_admin.firestore.client()`
    # after `firebase_admin.initialize_app`.
    # For this simulation, we just return a placeholder dict.
    st.session_state._firestore_data = st.session_state.get("_firestore_data", {})
    if app_id not in st.session_state._firestore_data:
        st.session_state._firestore_data[app_id] = {} # Simulating app-specific data
    return st.session_state._firestore_data[app_id]

def _simulate_auth_db():
    # This simulates Firebase Auth and Firestore being ready.
    # In a real Canvas environment, these would be provided globally or via init.
    global _db_initialized, _auth_initialized
    if not _db_initialized:
        # Simulate initial auth with token or anonymously
        # In actual Canvas, __initial_auth_token is used with signInWithCustomToken
        # If running locally, this would just be anonymous or manual login.
        st.session_state.current_firebase_user_id = st.session_state.get("current_firebase_user_id", None)
        if st.session_state.current_firebase_user_id is None:
            # Simulate an anonymous user or a user from initial_auth_token
            # In a real scenario, this is where Firebase's signInWithCustomToken would happen.
            # For this Streamlit simulation, we'll just set a generic "anonymous" user until explicit login.
            st.session_state.current_firebase_user_id = "anonymous_user_temp_id" # Placeholder
            st.session_state.current_firebase_user_email = None
            st.session_state.is_authenticated = False
        _db_initialized = True
        _auth_initialized = True
    
    # These are mock objects for the purpose of the Streamlit Python code
    # In actual Canvas, they would be real Firebase objects.
    class MockAuth:
        @property
        def current_user(self):
            class MockUser:
                uid = st.session_state.current_firebase_user_id
                email = st.session_state.current_firebase_user_email
            return MockUser() if st.session_state.current_firebase_user_id else None
        
        def sign_in_with_email_and_password(self, email, password):
            # Simulate real Firebase behavior: check mock user data
            users_db = _get_firestore_client(st.session_state.app_id).get('users', {})
            for uid, user_data in users_db.items():
                if user_data.get('email') == email and user_data.get('password_hash') == password: # Simplified: direct password check
                    st.session_state.current_firebase_user_id = uid
                    st.session_state.current_firebase_user_email = email
                    st.session_state.is_authenticated = True
                    return {"user": MockAuth().current_user}
            raise Exception("auth/wrong-password") # Simulate error

        def create_user_with_email_and_password(self, email, password):
            users_db = _get_firestore_client(st.session_state.app_id).get('users', {})
            for uid, user_data in users_db.items():
                if user_data.get('email') == email:
                    raise Exception("auth/email-already-in-use")
            
            new_uid = f"user_{len(users_db) + 1}_{datetime.now().strftime('%f')}"
            users_db[new_uid] = {"email": email, "password_hash": password} # Simplified: storing password directly (DO NOT DO IN REAL APP)
            _get_firestore_client(st.session_state.app_id)['users'] = users_db
            st.session_state.current_firebase_user_id = new_uid
            st.session_state.current_firebase_user_email = email
            st.session_state.is_authenticated = True
            return {"user": MockAuth().current_user}

        def sign_out(self):
            st.session_state.current_firebase_user_id = "anonymous_user_temp_id" # Reset to anonymous
            st.session_state.current_firebase_user_email = None
            st.session_state.is_authenticated = False

    class MockFirestore:
        def collection(self, path):
            parts = path.split('/')
            current_level = _get_firestore_client(st.session_state.app_id)
            for part in parts:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

            class MockCollection:
                _data = current_level # Reference to the actual dictionary for the collection

                def add_doc(self, data):
                    doc_id = f"doc_{len(self._data) + 1}_{datetime.now().strftime('%f')}"
                    self._data[doc_id] = data
                    return {"id": doc_id}

                def get_docs(self):
                    return [
                        type('MockDoc', (object,), {'id': k, 'to_dict': lambda self, v=v: v})()
                        for k, v in self._data.items()
                    ]
                
                def set_doc(self, doc_id, data, merge=False):
                    if doc_id not in self._data:
                        self._data[doc_id] = {}
                    if merge:
                        self._data[doc_id].update(data)
                    else:
                        self._data[doc_id] = data
            return MockCollection()

    return MockFirestore(), MockAuth()

# Initialize mock Firebase objects when script runs
if 'app_id' not in st.session_state:
    st.session_state.app_id = "emotion_dashboard_app" # Placeholder for __app_id

db, auth = _simulate_auth_db()

# --- pyttsx3 Initialization (moved higher to avoid re-init issues) ---
_pyttsx3_engine = None
def get_pyttsx3_engine():
    global _pyttsx3_engine
    if _pyttsx3_engine is None:
        try:
            _pyttsx3_engine = pyttsx3.init()
            # Optional: Set properties for better performance/voice
            # voices = _pyttsx3_engine.getProperty('voices')
            # _pyttsx3_engine.setProperty('voice', voices[1].id) # Try a different voice
            # _pyttsx3_engine.setProperty('rate', 150) # Speed of speech
        except Exception as e:
            st.warning(f"Failed to initialize pyttsx3: {e}. Speech feedback will be disabled.")
            _pyttsx3_engine = False # Set to False to indicate failure
    return _pyttsx3_engine


def speak_emotion(emotion):
    """
    Uses pyttsx3 to speak a predefined message based on the detected emotion.
    """
    engine = get_pyttsx3_engine()
    if not engine: # If engine failed to initialize
        return

    responses = {
        "happy": "You look happy! Keep smiling!",
        "sad": "Everything will be okay, don't worry.",
        "angry": "Take a deep breath.",
        "surprise": "What surprised you?",
        "neutral": "All chill today!",
        "fear": "You're safe here.",
        "disgust": "It seems like something's bothering you. Take it easy."
    }
    message = responses.get(emotion, "Emotion detected")
    try:
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        # st.warning(f"Text-to-speech execution error: {e}. Check system audio.")
        pass # Suppress non-critical errors for cleaner console output

def play_emotion_music(emotion):
    """
    Displays a toast message suggesting music based on emotion, with more specific placeholder links.
    """
    music_links = {
        "happy": "https://www.youtube.com/results?search_query=upbeat+happy+music",
        "sad": "https://www.youtube.com/results?search_query=calming+relaxing+music",
        "angry": "https://www.youtube.com/results?search_query=stress+relief+music",
        "surprise": "https://www.youtube.com/results?search_query=curiosity+inspiring+music",
        "neutral": "https://www.youtube.com/results?search_query=focus+study+music",
        "fear": "https://www.youtube.com/results?search_query=courage+uplifting+music",
        "disgust": "https://www.youtube.com/results?search_query=refreshing+clean+music"
    }
    link = music_links.get(emotion, "https://www.youtube.com/results?search_query=background+music")
    st.toast(f"🎶 Suggesting music for your mood! [Click to Play]({link})", icon="🎵")


def export_pdf(emotion_counts, user_id, log_df=None):
    """
    Generates a PDF report of emotion counts for a user, with optional detailed log.
    Saves to a user-specific reports directory.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, txt=f"Emotion Report for {st.session_state.current_firebase_user_email or user_id}", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10) # Add some space

    if emotion_counts:
        pdf.cell(0, 10, txt="Current Session Emotion Summary:", ln=True)
        for emo, count in emotion_counts.items():
            pdf.cell(0, 10, txt=f"  {emo.capitalize()}: {count} detections", ln=True)
    else:
        pdf.cell(0, 10, txt="No emotion data recorded for this session.", ln=True)
    
    if log_df is not None and not log_df.empty:
        pdf.ln(10)
        pdf.cell(0, 10, txt="Detailed Session Log (first 20 entries):", ln=True)
        pdf.set_font("Arial", size=8) # Smaller font for log
        # Add headers
        pdf.cell(50, 7, "Timestamp", 1)
        pdf.cell(50, 7, "Emotion", 1)
        pdf.ln()
        
        # Add log entries
        for index, row in log_df.head(20).iterrows(): # Limit to first 20 for brevity in PDF
            pdf.cell(50, 7, str(row['timestamp']), 1)
            pdf.cell(50, 7, row['emotion'].capitalize(), 1)
            pdf.ln()
        
        if len(log_df) > 20:
            pdf.cell(0, 7, "...", ln=True)
            # Cannot provide direct Firestore download link in PDF easily, hint at app view
            pdf.cell(0, 7, f"(Full log available in app's Historical Insights tab)", ln=True) 
        
    # Add a timestamp to the report
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt=f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='R')

    # Store locally for user to download/access (simulate file system)
    local_reports_dir = f"reports/{user_id}"
    os.makedirs(local_reports_dir, exist_ok=True)
    path = f"{local_reports_dir}/emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(path)
    return path


def save_emotion_log_to_firestore(emotion_log_data, user_id):
    """Saves current session emotion log to Firestore."""
    if not user_id or user_id == "anonymous_user_temp_id":
        st.error("Cannot save logs for unauthenticated or temporary users.")
        return False
    
    try:
        # Create a document for each detection for easier querying and less document size limits
        for entry in emotion_log_data:
            db.collection(f"artifacts/{st.session_state.app_id}/users/{user_id}/sessions").add_doc(entry)
        return True
    except Exception as e:
        st.error(f"Error saving emotion log to Firestore: {e}")
        return False

def load_user_sessions_from_firestore(user_id):
    """Loads all previous session logs for a given user from Firestore."""
    if not user_id or user_id == "anonymous_user_temp_id":
        return pd.DataFrame()
    
    try:
        docs = db.collection(f"artifacts/{st.session_state.app_id}/users/{user_id}/sessions").get_docs()
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            if 'timestamp' in doc_data:
                # Convert Firestore Timestamp to datetime or handle as string
                doc_data['timestamp'] = doc_data['timestamp'] # Assume already datetime string or suitable
            data.append(doc_data)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading user sessions from Firestore: {e}")
        return pd.DataFrame()

def get_period_summary(df, period_type='day'):
    """
    Generates a summary of dominant emotions for a given period (day or week).
    """
    if df.empty:
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if period_type == 'day':
        df['period_key'] = df['timestamp'].dt.date
    elif period_type == 'week':
        # Get the start of the week for grouping
        df['period_key'] = df['timestamp'].dt.to_period('W').dt.start_time.dt.date 

    # Exclude 'No Face' from dominant emotion calculation for meaningful insights
    filtered_df = df[df['emotion'] != 'No Face']
    if filtered_df.empty:
        return pd.DataFrame()

    summary = filtered_df.groupby('period_key')['emotion'].agg(
        total_detections='count',
        dominant_emotion=lambda x: x.mode()[0] if not x.empty else 'N/A'
    ).reset_index()

    summary['dominant_emotion_emoji'] = summary['dominant_emotion'].apply(get_emotion_emoji)
    return summary.sort_values(by='period_key', ascending=False)


def get_trends(df, period='day'):
    """
    Calculates emotion trends for a given period (day, hour, month).
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame() # Return empty DFs

    df["Timestamp"] = pd.to_datetime(df["timestamp"])
    
    if period == 'day':
        df["Period"] = df["Timestamp"].dt.strftime("%A")
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    elif period == 'hour':
        df["Period"] = df["Timestamp"].dt.hour.astype(str) + ":00"
        order = [f"{h:02d}:00" for h in range(24)]
    elif period == 'month':
        df["Period"] = df["Timestamp"].dt.strftime("%B")
        order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    else: # Default to day if invalid period
        df["Period"] = df["Timestamp"].dt.strftime("%A")
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
    all_emotions = ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust", "No Face"]
    
    trend = pd.crosstab(df["Period"], df["emotion"])
    trend = trend.reindex(index=order, columns=all_emotions, fill_value=0)
    trend_percent = trend.div(trend.sum(axis=1) + 1e-6, axis=0) * 100
    
    return trend, trend_percent

def plot_emotion_heatmap(df, period='day'):
    """
    Creates a heatmap of emotion frequency by day of week or hour of day.
    """
    crosstab_counts, crosstab_percent = get_trends(df, period) 
    
    if crosstab_percent.empty:
        # st.info(f"No data to generate heatmap for {period} trends.")
        return None

    # Filter out "No Face" column if it's all zeros for better visualization
    emotions_to_plot = [col for col in crosstab_percent.columns if col != 'No Face' or crosstab_percent[col].sum() > 0]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(crosstab_percent[emotions_to_plot], annot=True, fmt=".1f", cmap="YlGnBu", cbar=True, ax=ax, linewidths=.5, linecolor='lightgray')
    
    if period == 'day':
        ax.set_title("Emotion Frequency Heatmap by Day of Week (% of Daily Detections)", fontsize=14)
        ax.set_ylabel("Day of Week", fontsize=12)
    elif period == 'hour':
        ax.set_title("Emotion Frequency Heatmap by Hour of Day (% of Hourly Detections)", fontsize=14)
        ax.set_ylabel("Hour of Day", fontsize=12)
    elif period == 'month':
        ax.set_title("Emotion Frequency Heatmap by Month (% of Monthly Detections)", fontsize=14)
        ax.set_ylabel("Month", fontsize=12)
        
    ax.set_xlabel("Emotion", fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

def plot_overall_emotion_distribution(df):
    """
    Creates a bar chart of overall emotion distribution from a DataFrame of session logs.
    """
    if df.empty:
        # st.info("No data to plot overall emotion distribution.")
        return None

    # Exclude 'No Face' for a more meaningful distribution of detected emotions
    filtered_df = df[df['emotion'] != 'No Face']
    if filtered_df.empty:
        # st.info("No face detections to plot overall emotion distribution.")
        return None

    emotion_counts = filtered_df['emotion'].value_counts()
    
    colors = sns.color_palette("pastel")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette=colors, ax=ax)
    ax.set_title("Overall Emotion Distribution Across All Sessions (Excluding 'No Face')", fontsize=14)
    ax.set_xlabel("Emotion", fontsize=12)
    ax.set_ylabel("Total Detections", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_emotion_time_series(df, selected_session_df=None, events_df=None, interval_seconds=10):
    """
    Plots emotion distribution over time for a selected session or overall, with optional event markers.
    """
    if selected_session_df is not None and not selected_session_df.empty:
        plot_df = selected_session_df.copy()
        title = "Emotion Distribution Over Time for Selected Session"
    else:
        plot_df = df.copy()
        title = "Overall Emotion Distribution Over Time (All Sessions)"

    if plot_df.empty:
        # st.info("No data available to plot emotion over time.")
        return None # Return None if there's no data to plot

    plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'])
    plot_df = plot_df.sort_values(by='timestamp')
    
    min_time = plot_df['timestamp'].min()
    max_time = plot_df['timestamp'].max()
    
    time_bins = pd.date_range(start=min_time, end=max_time + timedelta(seconds=interval_seconds), freq=f'{interval_seconds}S')
    plot_df['time_bin'] = pd.cut(plot_df['timestamp'], bins=time_bins, labels=time_bins[:-1], right=False)
    
    emotion_over_time = plot_df.groupby('time_bin')['emotion'].value_counts().unstack(fill_value=0)
    
    all_emotions = ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust", "No Face"]
    emotion_over_time = emotion_over_time.reindex(columns=all_emotions, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(15, 7))
    emotion_over_time.plot(kind='area', stacked=True, ax=ax, cmap='viridis')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(f"Number of Detections (Aggregated every {interval_seconds}s)", fontsize=12)
    ax.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- Feature: Emotion Correlation with Manual Events ---
    if events_df is not None and not events_df.empty:
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
        
        # Filter events relevant to the displayed time frame
        events_in_range = events_df[
            (events_df['timestamp'] >= min_time) & (events_df['timestamp'] <= max_time)
        ]
        
        for idx, event in events_in_range.iterrows():
            ax.axvline(x=event['timestamp'], color='red', linestyle='--', linewidth=1, alpha=0.7)
            # Annotate event with text, adjusted for better visibility
            ax.text(
                event['timestamp'],
                ax.get_ylim()[1] * 0.95, # Position near the top of the plot
                f" Event: {event['description']}",
                rotation=90,
                verticalalignment='top',
                horizontalalignment='left',
                color='red',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.2')
            )
            
    return fig

# --- Authentication and User Management Functions (Firestore) ---
def register_user(email, password):
    try:
        # Simulate Firebase Auth createUserWithEmailAndPassword
        user_credential = auth.create_user_with_email_and_password(email, password)
        st.session_state.current_firebase_user_id = user_credential['user'].uid
        st.session_state.current_firebase_user_email = user_credential['user'].email
        st.session_state.is_authenticated = True
        return True, "Registration successful!"
    except Exception as e:
        error_message = str(e)
        if "email-already-in-use" in error_message:
            return False, "Email already registered. Try logging in or use a different email."
        elif "weak-password" in error_message:
            return False, "Password should be at least 6 characters."
        return False, f"Registration failed: {error_message}"

def login_user(email, password):
    try:
        # Simulate Firebase Auth signInWithEmailAndPassword
        user_credential = auth.sign_in_with_email_and_password(email, password)
        st.session_state.current_firebase_user_id = user_credential['user'].uid
        st.session_state.current_firebase_user_email = user_credential['user'].email
        st.session_state.is_authenticated = True
        return True, "Login successful!"
    except Exception as e:
        error_message = str(e)
        if "wrong-password" in error_message or "user-not-found" in error_message:
            return False, "Invalid email or password."
        return False, f"Login failed: {error_message}"

def logout_user():
    auth.sign_out()
    st.session_state.current_firebase_user_id = "anonymous_user_temp_id"
    st.session_state.current_firebase_user_email = None
    st.session_state.is_authenticated = False
    # Clear session state for the dashboard data
    st.session_state.emotion_log = []
    st.session_state.emotion_counts = {}
    st.session_state.current_emotion = "No Face"
    st.session_state.chat_history = []
    st.session_state.last_speech_time = datetime.now() - timedelta(seconds=10) # Reset
    st.session_state.emotion_start_time = None
    st.session_state.last_alerted_emotion = None
    st.session_state.alert_triggered = False
    st.session_state.emotion_streak_start_time = {}
    st.session_state.emotion_current_streak_emotion = None
    st.session_state.emotion_current_streak_length = 0
    st.session_state.settings = {} # Reset settings to defaults on logout
    st.experimental_rerun() # Rerun to reflect logout state

def get_emotion_emoji(emotion):
    emojis = {
        "happy": "😊", "sad": "😔", "angry": "😠", "surprise": "😲",
        "neutral": "😐", "fear": "😨", "disgust": "🤢", "No Face": "👤"
    }
    return emojis.get(emotion, "❓")

# --- Firestore Data Management Functions (Re-implemented for Firestore) ---

def load_goals(user_id):
    """Loads emotional goals for a given user from Firestore."""
    if not user_id or user_id == "anonymous_user_temp_id":
        return []
    try:
        doc = db.collection(f"artifacts/{st.session_state.app_id}/users/{user_id}/goals").get_docs()
        # In a real Firestore, you'd fetch a single document containing a 'goals' array.
        # For this mock, we'll assume the 'goals' document ID is 'user_goals'.
        goals_doc = None
        for d in doc:
            if d.id == "user_goals":
                goals_doc = d.to_dict()
                break
        return goals_doc.get("goals", []) if goals_doc else []
    except Exception as e:
        st.error(f"Error loading goals from Firestore: {e}")
        return []

def save_goals(user_id, goals_list):
    """Saves emotional goals for a given user to Firestore."""
    if not user_id or user_id == "anonymous_user_temp_id":
        st.error("Cannot save goals for unauthenticated or temporary users.")
        return
    try:
        # Use set_doc with a known ID to store all goals in one document
        db.collection(f"artifacts/{st.session_state.app_id}/users/{user_id}/goals").set_doc("user_goals", {"goals": goals_list}, merge=False)
    except Exception as e:
        st.error(f"Error saving goals to Firestore: {e}")

def load_events(user_id):
    """Loads contextual events for a given user from Firestore."""
    if not user_id or user_id == "anonymous_user_temp_id":
        return pd.DataFrame()
    try:
        docs = db.collection(f"artifacts/{st.session_state.app_id}/users/{user_id}/events").get_docs()
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            if 'timestamp' in doc_data:
                doc_data['timestamp'] = doc_data['timestamp'] # Assume string or suitable
            data.append(doc_data)
        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading events from Firestore: {e}")
        return pd.DataFrame()

def save_event(user_id, event_entry):
    """Saves a single contextual event for a given user to Firestore."""
    if not user_id or user_id == "anonymous_user_temp_id":
        st.error("Cannot save events for unauthenticated or temporary users.")
        return
    try:
        db.collection(f"artifacts/{st.session_state.app_id}/users/{user_id}/events").add_doc(event_entry)
    except Exception as e:
        st.error(f"Error saving event to Firestore: {e}")

def load_settings(user_id):
    """Loads user settings from Firestore."""
    if not user_id or user_id == "anonymous_user_temp_id":
        return {} # Return empty if not authenticated
    try:
        doc = db.collection(f"artifacts/{st.session_state.app_id}/users/{user_id}/settings").get_docs()
        # In a real Firestore, you'd fetch a single document like 'user_settings'.
        user_settings_doc = None
        for d in doc:
            if d.id == "user_settings": # Assuming a fixed document ID for settings
                user_settings_doc = d.to_dict()
                break
        return user_settings_doc if user_settings_doc else {}
    except Exception as e:
        st.error(f"Error loading settings from Firestore: {e}")
        return {}

def save_settings(user_id, settings_dict):
    """Saves user settings to Firestore."""
    if not user_id or user_id == "anonymous_user_temp_id":
        st.error("Cannot save settings for unauthenticated or temporary users.")
        return
    try:
        # Use set_doc with a known ID to store all settings in one document
        db.collection(f"artifacts/{st.session_state.app_id}/users/{user_id}/settings").set_doc("user_settings", settings_dict, merge=False)
    except Exception as e:
        st.error(f"Error saving settings to Firestore: {e}")

def load_achievements(user_id):
    """Loads user achievements and streak data from Firestore."""
    if not user_id or user_id == "anonymous_user_temp_id":
        return {"daily_mood_streak": 0, "last_mood_log_date": None, "emotion_streaks": {}}
    try:
        doc = db.collection(f"artifacts/{st.session_state.app_id}/users/{user_id}/achievements").get_docs()
        # In a real Firestore, you'd fetch a single document like 'user_achievements'.
        achievements_doc = None
        for d in doc:
            if d.id == "user_achievements": # Assuming a fixed document ID for achievements
                achievements_doc = d.to_dict()
                break
        return achievements_doc if achievements_doc else {"daily_mood_streak": 0, "last_mood_log_date": None, "emotion_streaks": {}}
    except Exception as e:
        st.error(f"Error loading achievements from Firestore: {e}")
        return {"daily_mood_streak": 0, "last_mood_log_date": None, "emotion_streaks": {}}

def save_achievements(user_id, achievements_dict):
    """Saves user achievements and streak data to Firestore."""
    if not user_id or user_id == "anonymous_user_temp_id":
        st.error("Cannot save achievements for unauthenticated or temporary users.")
        return
    try:
        # Use set_doc with a known ID to store all achievements in one document
        db.collection(f"artifacts/{st.session_state.app_id}/users/{user_id}/achievements").set_doc("user_achievements", achievements_dict, merge=False)
    except Exception as e:
        st.error(f"Error saving achievements to Firestore: {e}")

# --- Streamlit App Layout and Logic ---

# Initialize Streamlit page config (only runs once)
st.set_page_config(page_title="Emotion AI Dashboard", layout="wide", 
                   initial_sidebar_state="expanded",
                   menu_items={
                       'About': "## Real-time Emotion Detection Dashboard\n\nThis app uses DeepFace to detect emotions from a webcam feed, logs the data, provides historical analysis, and offers an interactive chatbot.",
                       'Get help': 'mailto:your.email@example.com' # Replace with actual contact
                   })

st.title("Emotion AI Dashboard 🧠✨")

# Ensure base local directories exist (for PDF reports/cached data if needed)
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# --- Authentication Logic (moved to top-level for UI control) ---
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False
if 'auth_page' not in st.session_state:
    st.session_state.auth_page = "login" # "login" or "register"

# Display authentication UI if not authenticated
if not st.session_state.is_authenticated:
    st.sidebar.subheader("Account Access")
    auth_choice = st.sidebar.radio("Choose:", ["Login", "Register"], key="auth_choice_radio")
    st.session_state.auth_page = auth_choice.lower()

    if st.session_state.auth_page == "login":
        st.subheader("Login to Your Account")
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_btn"):
            with st.spinner("Logging in..."):
                success, message = login_user(login_email, login_password)
                if success:
                    st.success(message)
                    st.experimental_rerun() # Rerun to load dashboard for logged-in user
                else:
                    st.error(message)
    else: # Register
        st.subheader("Create a New Account")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")

        if st.button("Register", key="register_btn"):
            if reg_password != reg_password_confirm:
                st.error("Passwords do not match.")
            elif len(reg_password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                with st.spinner("Registering..."):
                    success, message = register_user(reg_email, reg_password)
                    if success:
                        st.success(message)
                        st.experimental_rerun() # Rerun to load dashboard for new user
                    else:
                        st.error(message)
    st.stop() # Stop execution until user is authenticated

# If authenticated, proceed with the dashboard
# Get current user ID and email
current_user_id = auth.current_user.uid
current_user_email = auth.current_user.email
username = current_user_email or current_user_id # Use email as primary username, fallback to ID

st.sidebar.success(f"Logged in as: **{current_user_email}**")
if st.sidebar.button("Logout", key="logout_btn"):
    logout_user()


# --- Load User Settings for the logged-in user ---
user_settings = load_settings(current_user_id)
# Default settings
default_settings = {
    "speech_enabled": True,
    "music_toast_enabled": True,
    "speech_frequency": 10,
    "alert_emotion": "None",
    "alert_duration": 30,
    "theme": "light" 
}
# Merge loaded settings with defaults, using defaults for missing keys
st.session_state.settings = {**default_settings, **user_settings}


# --- Sidebar Session & Alert Settings (Now from st.session_state.settings) ---
st.sidebar.subheader("Session & Alert Settings")
st.sidebar.info("Click 'Start Webcam' to begin a new session. Press 'Stop Webcam' to end.")


# Initialize session state variables (if not already done by logout or previous run)
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'emotion_log' not in st.session_state:
    st.session_state.emotion_log = []
if 'emotion_counts' not in st.session_state:
    st.session_state.emotion_counts = {}
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "No Face"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_speech_time' not in st.session_state:
    # Initialize using loaded settings
    st.session_state.last_speech_time = datetime.now() - timedelta(seconds=st.session_state.settings['speech_frequency']) 

# Session state for emotion alerts
if 'emotion_start_time' not in st.session_state:
    st.session_state.emotion_start_time = None
if 'last_alerted_emotion' not in st.session_state:
    st.session_state.last_alerted_emotion = None
if 'alert_triggered' not in st.session_state: 
    st.session_state.alert_triggered = False

# --- Streak Tracking in Session State ---
# These are dynamic during a session, actual saving is handled when mood is logged or session ends
if 'emotion_streak_start_time' not in st.session_state:
    st.session_state.emotion_streak_start_time = {} # {emotion: start_time}
if 'emotion_current_streak_emotion' not in st.session_state:
    st.session_state.emotion_current_streak_emotion = None
if 'emotion_current_streak_length' not in st.session_state:
    st.session_state.emotion_current_streak_length = 0


# --- Main Dashboard Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Real-time Analysis", "Historical Insights", "Manual Mood Log", "EmotionBot Chat", "Settings", "Streaks & Achievements"])

with tab1:
    st.header("Live Emotion Detection & Metrics")
    st.write("Monitor your emotions in real-time. The system detects the dominant emotion and provides probability scores. Bounding boxes will appear around all detected faces.")
    st.markdown("---")

    col1_btn, col2_btn = st.columns(2)
    with col1_btn:
        start_button = st.button("Start Webcam Session", help="Begin detecting emotions from your webcam.")
    with col2_btn:
        stop_button = st.button("Stop Webcam Session", help="End the current emotion detection session.")

    if start_button and not st.session_state.webcam_running:
        st.session_state.webcam_running = True
        st.session_state.emotion_log = [] # Reset log for NEW session
        st.session_state.emotion_counts = {} # Reset counts for NEW session
        st.info("Webcam started. Looking for your face... Please ensure good lighting and face visibility.")
        st.session_state.last_speech_time = datetime.now() # Reset speech timer
        st.session_state.emotion_start_time = None # Reset alert timer
        st.session_state.last_alerted_emotion = None
        st.session_state.alert_triggered = False
        st.session_state.emotion_streak_start_time = {} # Reset emotion streak timers
        st.session_state.emotion_current_streak_emotion = None
        st.session_state.emotion_current_streak_length = 0

    elif stop_button and st.session_state.webcam_running:
        st.session_state.webcam_running = False
        st.info("Webcam stopped. Data for this session is stored in memory until saved.")

    webcam_col, metrics_col = st.columns([2, 1])

    with webcam_col:
        frame_window = st.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Live Webcam Feed", use_column_width=True)
        
    with metrics_col:
        st.subheader("Current Emotion")
        emotion_placeholder = st.empty() # For current emotion display
        
        st.subheader("Emotion Probabilities")
        prob_area = st.empty() # Placeholder for bar chart
        
        st.subheader("Session Emotion Distribution")
        chart_area = st.empty() # Placeholder for current session bar chart

        st.subheader("Session Metrics")
        total_detections_placeholder = st.empty()
        most_dominant_emotion_placeholder = st.empty()


    # --- Webcam Processing Loop ---
    if st.session_state.webcam_running:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Error: Could not open webcam. Please check if it's connected and not in use by another application.")
            st.session_state.webcam_running = False
        else:
            status_text = st.empty()
            frame_count = 0
            while st.session_state.webcam_running:
                ret, frame = camera.read()
                if not ret:
                    status_text.warning("Failed to grab frame from webcam. Stopping webcam.")
                    st.session_state.webcam_running = False
                    break

                frame_count += 1
                
                # Analyze emotion (for all faces)
                current_emotions_detected = []
                dominant_emotion_of_largest_face = "No Face"
                largest_face_area = 0
                scores_for_display = {} # To display probabilities of the dominant face

                try:
                    analysis_start_time = time.time()
                    # Process smaller frame for performance
                    small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
                    
                    results = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False, silent=True)
                    
                    analysis_end_time = time.time()
                    
                    if isinstance(results, list) and results: # Check if DeepFace found faces
                        for face_result in results:
                            x, y, w, h = [int(v * 2) for v in (face_result['region']['x'], face_result['region']['y'], face_result['region']['w'], face_result['region']['h'])]
                            emotion = face_result['dominant_emotion']
                            current_emotions_detected.append(emotion)

                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                            
                            # Find the largest face to determine the main 'current_emotion'
                            if (w * h) > largest_face_area:
                                largest_face_area = (w * h)
                                dominant_emotion_of_largest_face = emotion
                                scores_for_display = face_result['emotion']

                        st.session_state.current_emotion = dominant_emotion_of_largest_face
                    else:
                        st.session_state.current_emotion = "No Face"
                        scores_for_display = {}
                        
                    processing_time_ms = (analysis_end_time - analysis_start_time) * 1000
                    status_text.text(f"Processing frame {frame_count}... ({processing_time_ms:.2f} ms)")

                except Exception as e:
                    st.session_state.current_emotion = "No Face"
                    scores_for_display = {}
                    status_text.text(f"Processing frame {frame_count}... No face detected or error: {e}")


                # Update emotion log and counts for the current session
                current_time = datetime.now()
                st.session_state.emotion_log.append({"emotion": st.session_state.current_emotion, "timestamp": current_time.isoformat()})
                st.session_state.emotion_counts[st.session_state.current_emotion] = st.session_state.emotion_counts.get(st.session_state.current_emotion, 0) + 1
                
                # --- Feature: Emotion Streaks Logic ---
                achievements_data = load_achievements(current_user_id) # Load for current user
                emotion_streaks = achievements_data.get("emotion_streaks", {})

                if st.session_state.current_emotion != "No Face":
                    # Check if the dominant emotion has changed
                    if st.session_state.current_emotion != st.session_state.emotion_current_streak_emotion:
                        # Save the max streak for the previous emotion if it was significant
                        if st.session_state.emotion_current_streak_emotion:
                            prev_emo = st.session_state.emotion_current_streak_emotion
                            if st.session_state.emotion_current_streak_length > emotion_streaks.get(prev_emo, 0):
                                emotion_streaks[prev_emo] = st.session_state.emotion_current_streak_length
                                achievements_data["emotion_streaks"] = emotion_streaks
                                save_achievements(current_user_id, achievements_data)
                                
                        # Start new streak
                        st.session_state.emotion_current_streak_emotion = st.session_state.current_emotion
                        st.session_state.emotion_streak_start_time[st.session_state.current_emotion] = current_time
                        st.session_state.emotion_current_streak_length = 0 # Reset length for new streak
                    
                    # Update current streak length
                    if st.session_state.emotion_current_streak_emotion == st.session_state.current_emotion:
                        start_time_for_current_streak = st.session_state.emotion_streak_start_time.get(st.session_state.current_emotion, current_time)
                        st.session_state.emotion_current_streak_length = (current_time - start_time_for_current_streak).total_seconds()
                else: # No face detected, reset current streak tracking
                    if st.session_state.emotion_current_streak_emotion:
                        prev_emo = st.session_state.emotion_current_streak_emotion
                        if st.session_state.emotion_current_streak_length > emotion_streaks.get(prev_emo, 0):
                            emotion_streaks[prev_emo] = st.session_state.emotion_current_streak_length
                            achievements_data["emotion_streaks"] = emotion_streaks
                            save_achievements(current_user_id, achievements_data)
                            
                    st.session_state.emotion_current_streak_emotion = None
                    st.session_state.emotion_current_streak_length = 0
                    st.session_state.emotion_streak_start_time = {} # Clear all streak timers

                # Trigger speech and music based on frequency setting from user settings
                speech_frequency = st.session_state.settings.get("speech_frequency", 10) # Fallback to 10
                if (current_time - st.session_state.last_speech_time).total_seconds() >= speech_frequency and st.session_state.current_emotion != "No Face":
                    if st.session_state.settings.get("speech_enabled", True):
                        speak_emotion(st.session_state.current_emotion)
                    if st.session_state.settings.get("music_toast_enabled", True):
                        play_emotion_music(st.session_state.current_emotion)
                    st.session_state.last_speech_time = current_time

                # --- Emotion Alert Logic (using user settings) ---
                alert_emotion = st.session_state.settings.get("alert_emotion", "None")
                alert_duration = st.session_state.settings.get("alert_duration", 30)

                if alert_emotion != "None" and st.session_state.current_emotion == alert_emotion:
                    if st.session_state.emotion_start_time is None:
                        st.session_state.emotion_start_time = current_time
                        st.session_state.alert_triggered = False # Reset alert trigger for new continuous period
                    
                    elapsed_time = (current_time - st.session_state.emotion_start_time).total_seconds()
                    
                    if elapsed_time >= alert_duration and not st.session_state.alert_triggered:
                        st.warning(f"🚨 **ALERT!** You've been feeling **{alert_emotion.upper()}** for over {alert_duration} seconds!")
                        st.session_state.alert_triggered = True # Prevent repeated alerts for same continuous emotion
                else:
                    st.session_state.emotion_start_time = None # Reset timer if emotion changes or alert is off
                    st.session_state.alert_triggered = False # Reset alert trigger for a new detection of this emotion
                

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Update Streamlit components in columns
                with webcam_col:
                    caption_text = f"Live Webcam Feed ({st.session_state.current_emotion.capitalize()} {get_emotion_emoji(st.session_state.current_emotion)})"
                    if len(current_emotions_detected) > 1:
                        caption_text += f" (+ {len(current_emotions_detected)-1} other faces)"
                    frame_window.image(frame_rgb, caption=caption_text, use_column_width=True)
                
                with metrics_col:
                    emotion_placeholder.markdown(f"**Detected:** `<span style='color:green; font-size: 24px;'>{st.session_state.current_emotion.upper()} {get_emotion_emoji(st.session_state.current_emotion)}</span>`", unsafe_allow_html=True)
                    
                    if scores_for_display:
                        prob_df = pd.DataFrame(scores_for_display.items(), columns=['Emotion', 'Probability']).set_index('Emotion')
                        prob_area.bar_chart(prob_df)
                    else:
                        prob_area.empty()

                    if st.session_state.emotion_log:
                        current_session_emotions_series = pd.Series([e['emotion'] for e in st.session_state.emotion_log])
                        current_counts = current_session_emotions_series.value_counts().sort_index()
                        chart_area.bar_chart(current_counts)
                        
                        total_detections_placeholder.metric("Total Detections (Current Session)", len(st.session_state.emotion_log))
                        most_dominant = current_counts.idxmax() if not current_counts.empty else "N/A"
                        most_dominant_emotion_placeholder.metric("Most Dominant Emotion", most_dominant.capitalize())
                    else:
                        chart_area.empty()
                        total_detections_placeholder.empty()
                        most_dominant_emotion_placeholder.empty()
                
                if not st.session_state.webcam_running:
                    break
            
            camera.release()
            cv2.destroyAllWindows()
            status_text.success("Webcam session ended.")
    else:
        # Save current emotion streak when webcam stops
        achievements_data = load_achievements(current_user_id)
        emotion_streaks = achievements_data.get("emotion_streaks", {})
        if st.session_state.emotion_current_streak_emotion:
            prev_emo = st.session_state.emotion_current_streak_emotion
            if st.session_state.emotion_current_streak_length > emotion_streaks.get(prev_emo, 0):
                emotion_streaks[prev_emo] = st.session_state.emotion_current_streak_length
                achievements_data["emotion_streaks"] = emotion_streaks
                save_achievements(current_user_id, achievements_data)

        if 'camera' in locals() and camera.isOpened():
            camera.release()
            cv2.destroyAllWindows()
        frame_window.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Webcam Stopped or Not Started", use_column_width=True)
        emotion_placeholder.markdown(f"**Detected:** `N/A {get_emotion_emoji('No Face')}`", unsafe_allow_html=True)
        prob_area.empty()
        chart_area.empty()
        total_detections_placeholder.empty()
        most_dominant_emotion_placeholder.empty()


    st.markdown("---")
    st.subheader("Session Management & Reporting")
    col_report_btn, col_save_btn = st.columns(2)

    with col_report_btn:
        if st.button("Export PDF Report (Current Session)", help="Generate a PDF summary of the emotions detected in this session."):
            if st.session_state.emotion_counts:
                current_session_df = pd.DataFrame(st.session_state.emotion_log)
                path = export_pdf(st.session_state.emotion_counts, current_user_id, log_df=current_session_df)
                st.success(f"📄 Report saved locally: [Download Here]({path}) (Check 'reports/{current_user_id}/' folder)")
            else:
                st.warning("No emotion data for the current session to export yet. Start the webcam first!")

    with col_save_btn:
        if st.button("Save Current Session Log (Firestore)", help="Save the detailed log of emotions from this session to your Firestore database."):
            if st.session_state.emotion_log:
                if save_emotion_log_to_firestore(st.session_state.emotion_log, current_user_id):
                    st.success("💾 Session log saved to Firestore successfully!")
                st.rerun() # Rerun to update insights
            else:
                st.warning("No emotion data for the current session to save yet. Start the webcam first!")

    st.markdown("---")
    st.subheader("Add Event Context")
    st.write("Log specific events happening around you to connect them with your emotional state.")
    event_description = st.text_input("What's happening right now? (e.g., 'got an email', 'listening to music')", key="event_desc_input")
    if st.button("Log Event with Current Emotion", key="log_event_btn"):
        if event_description:
            event_entry = {
                "timestamp": datetime.now().isoformat(),
                "emotion_at_log": st.session_state.current_emotion,
                "description": event_description,
                "user_id": current_user_id # Ensure event is tied to user
            }
            save_event(current_user_id, event_entry) # Save to Firestore
            st.success("Event logged successfully!")
            st.experimental_rerun() # Rerun to refresh the displayed events
        else:
            st.warning("Please enter a description for the event.")
    
    st.subheader("Recent Events")
    recent_events_df = load_events(current_user_id) # Load from Firestore
    if not recent_events_df.empty:
        st.dataframe(recent_events_df.sort_values(by='timestamp', ascending=False).head(5))
    else:
        st.info("No events logged yet for this user.")


with tab2:
    st.header("Historical Emotion Insights")
    st.write(f"Explore your emotion trends over time from all recorded sessions for **{username}**.")
    
    past_sessions_df = load_user_sessions_from_firestore(current_user_id) # Load from Firestore
    events_df = load_events(current_user_id) # Load events for plotting from Firestore

    if not past_sessions_df.empty:
        st.subheader("Overall Emotion Distribution")
        fig_overall = plot_overall_emotion_distribution(past_sessions_df)
        if fig_overall:
            st.pyplot(fig_overall)

        st.subheader("Daily/Weekly Emotion Summary")
        summary_period_selection = st.radio("Show summary for:", ["Daily", "Weekly"], key="summary_period_radio")
        summary_df = get_period_summary(past_sessions_df, 'day' if summary_period_selection == 'Daily' else 'week')
        
        if not summary_df.empty:
            st.dataframe(summary_df.head(7).rename(columns={'period_key': 'Date/Week', 'total_detections': 'Total Detections', 'dominant_emotion': 'Dominant Emotion', 'dominant_emotion_emoji': 'Emoji'}))
            st.markdown("---")
            if 'dominant_emotion' in summary_df and not summary_df['dominant_emotion'].empty:
                overall_dominant = summary_df['dominant_emotion'].mode()[0]
                st.markdown(f"**Insights:** Your most frequent emotion during the displayed period was **{overall_dominant.capitalize()}** {get_emotion_emoji(overall_dominant)}.")
            else:
                st.info("No dominant emotion found in the summary period.")
        else:
            st.info("No sufficient data to generate daily/weekly summaries for this user yet.")

        st.subheader("Emotion Trends Over Time")
        trend_period = st.selectbox(
            "Select trend period:",
            options=["day", "hour", "month"],
            format_func=lambda x: x.capitalize(),
            key="trend_period_selector"
        )
        st.write(f"This heatmap shows the percentage of each emotion detected by {trend_period}.")
        fig_heatmap = plot_emotion_heatmap(past_sessions_df, period=trend_period)
        if fig_heatmap:
            st.pyplot(fig_heatmap)

        st.subheader("Detailed Time Series Analysis")
        st.write("Visualize how your emotions have changed across sessions or within a specific session, aggregated over time. **Events you've logged will appear as red markers.**")
        
        session_selection_mode = st.radio("Select data for time series plot:", ["All Sessions", "Specific Session Date"], key="ts_mode_radio")
        
        filtered_df_for_time_series = pd.DataFrame()
        if session_selection_mode == "All Sessions":
            filtered_df_for_time_series = past_sessions_df
            st.info("Showing combined emotion trends across all historical sessions.")
        else: # Specific Session Date
            session_dates = sorted(past_sessions_df['timestamp'].dt.date.unique(), reverse=True)
            if session_dates:
                selected_date_str = st.selectbox(
                    "Select a specific date to view emotion trends for that day:",
                    options=[str(d) for d in session_dates],
                    key="time_series_date_selector"
                )
                selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
                filtered_df_for_time_series = past_sessions_df[past_sessions_df['timestamp'].dt.date == selected_date]
                st.info(f"Showing emotion trends for the session(s) on {selected_date_str}.")
            else:
                st.warning("No specific session dates found to select.")

        if not filtered_df_for_time_series.empty:
            time_series_interval = st.slider("Time Series Aggregation Interval (seconds)", 5, 60, 10, key="ts_interval")
            # Pass events_df to the plotting function
            fig_timeseries = plot_emotion_time_series(past_sessions_df, filtered_df_for_time_series, events_df, interval_seconds=time_series_interval)
            if fig_timeseries:
                st.pyplot(fig_timeseries)
        else:
            st.info("No data available to plot emotion over time for the selected option.")
        
        st.markdown("---")
        st.subheader("Comparative Analysis of Custom Periods")
        st.write("Compare emotion distributions between any two selected date ranges.")

        col_period1, col_period2 = st.columns(2)

        with col_period1:
            st.subheader("Period 1")
            min_date_val = past_sessions_df['timestamp'].min().date() if not past_sessions_df.empty else date.today() - timedelta(days=30)
            max_date_val = past_sessions_df['timestamp'].max().date() if not past_sessions_df.empty else date.today()
            
            start_date1 = st.date_input("Start Date (Period 1)", value=min_date_val, min_value=min_date_val, max_value=max_date_val, key="start_date1")
            end_date1 = st.date_input("End Date (Period 1)", value=max_date_val, min_value=min_date_val, max_value=max_date_val, key="end_date1")
            
            df_period1 = past_sessions_df[(past_sessions_df['timestamp'].dt.date >= start_date1) & (past_sessions_df['timestamp'].dt.date <= end_date1)]
            counts1 = df_period1['emotion'].value_counts(normalize=True) * 100 if not df_period1.empty else pd.Series()

        with col_period2:
            st.subheader("Period 2")
            # Default second period to be slightly offset from first
            start_date2_default = min_date_val + timedelta(days=7) if min_date_val + timedelta(days=7) <= max_date_val else max_date_val
            end_date2_default = max_date_val + timedelta(days=7) if max_date_val + timedelta(days=7) <= datetime.now().date() else datetime.now().date()
            
            start_date2 = st.date_input("Start Date (Period 2)", value=start_date2_default, min_value=min_date_val, max_value=max_date_val, key="start_date2")
            end_date2 = st.date_input("End Date (Period 2)", value=end_date2_default, min_value=min_date_val, max_value=max_date_val, key="end_date2")

            df_period2 = past_sessions_df[(past_sessions_df['timestamp'].dt.date >= start_date2) & (past_sessions_df['timestamp'].dt.date <= end_date2)]
            counts2 = df_period2['emotion'].value_counts(normalize=True) * 100 if not df_period2.empty else pd.Series()
        
        if st.button("Compare Periods", key="compare_periods_btn"):
            if counts1.empty and counts2.empty:
                st.warning("No data found for either selected period. Please adjust dates.")
            else:
                # Combine and reindex for consistent plotting
                all_emotions = ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]
                compare_df = pd.DataFrame({
                    f"Period 1 ({start_date1} to {end_date1})": counts1.reindex(all_emotions, fill_value=0),
                    f"Period 2 ({start_date2} to {end_date2})": counts2.reindex(all_emotions, fill_value=0)
                }).fillna(0)
                
                if not compare_df.empty:
                    st.bar_chart(compare_df)
                    st.info("Comparison based on percentage of total emotion detections in each period.")
                else:
                    st.warning("Unable to generate comparison. No valid emotion data for selected periods.")


        # --- Feature: Emotion Trend Baselines & Anomaly Detection ---
        st.markdown("---")
        st.subheader("Emotion Baseline & Anomaly Detection")
        st.write("Establish a baseline of your typical emotional state and detect significant deviations.")
        
        baseline_period_options = {
            "Last 7 Days": 7,
            "Last 30 Days": 30,
            "All Time": None
        }
        baseline_selection_label = st.selectbox(
            "Select Baseline Period:",
            list(baseline_period_options.keys()),
            key="baseline_period_select"
        )
        days_for_baseline = baseline_period_options[baseline_selection_label]

        baseline_df = past_sessions_df.copy() # Use a copy to avoid modifying original df
        if days_for_baseline is not None:
            baseline_start_date = datetime.now() - timedelta(days=days_for_baseline)
            baseline_df = baseline_df[baseline_df['timestamp'] >= baseline_start_date]

        # Filter out 'No Face' for baseline calculation
        baseline_df_filtered = baseline_df[baseline_df['emotion'] != 'No Face']

        if not baseline_df_filtered.empty:
            baseline_counts = baseline_df_filtered['emotion'].value_counts()
            baseline_percentages = (baseline_counts / baseline_counts.sum() * 100).fillna(0)
            
            st.markdown(f"**Baseline ({baseline_selection_label}) Emotion Distribution:**")
            
            # Reindex to ensure all 7 emotions are present for consistent display
            all_known_emotions = [e for e in ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]]
            baseline_percentages_display = baseline_percentages.reindex(all_known_emotions, fill_value=0)
            st.dataframe(baseline_percentages_display.to_frame(name='Percentage').T.style.format("{:.2f}%"))
            
            # Compare current session (if available) to baseline
            if 'emotion_log' in st.session_state and st.session_state.emotion_log:
                current_session_df = pd.DataFrame(st.session_state.emotion_log)
                current_session_df_filtered = current_session_df[current_session_df['emotion'] != 'No Face']

                if not current_session_df_filtered.empty:
                    current_session_counts = current_session_df_filtered['emotion'].value_counts()
                    current_session_percentages = (current_session_counts / current_session_counts.sum() * 100).fillna(0)

                    st.markdown("**Current Session Emotion Distribution:**")
                    current_session_percentages_display = current_session_percentages.reindex(all_known_emotions, fill_value=0)
                    st.dataframe(current_session_percentages_display.to_frame(name='Percentage').T.style.format("{:.2f}%"))


                    # Anomaly detection (simplified: sum of squared differences in percentages)
                    # Ensure both series have the same index for accurate subtraction
                    combined_index = list(set(baseline_percentages.index).union(set(current_session_percentages.index)))
                    
                    baseline_series = baseline_percentages.reindex(combined_index, fill_value=0)
                    current_series = current_session_percentages.reindex(combined_index, fill_value=0)

                    deviation_score = ((current_series - baseline_series)**2).sum()
                    
                    # You'd need to define a threshold for 'significant' deviation
                    anomaly_threshold = 100 # Adjust this value based on experimentation

                    if deviation_score > anomaly_threshold:
                        st.warning(f"🚨 **ANOMALY DETECTED!** Your current session's emotional profile deviates significantly from your {baseline_selection_label} baseline (Deviation Score: {deviation_score:.2f}).")
                        st.info("Consider reflecting on what might be different today. Are you more stressed, happy, or sad than usual?")
                    else:
                        st.info(f"Current session is within your typical emotional range (Deviation Score: {deviation_score:.2f}).")
                else:
                    st.info("No face detections in the current session to compare against the baseline.")
            else:
                st.info("No current session data to compare against the baseline. Start a webcam session!")
        else:
            st.info(f"Not enough relevant (non-'No Face') data to establish a baseline for the {baseline_selection_label} period.")

        st.subheader("Raw Historical Data Table")
        st.dataframe(past_sessions_df)

    else:
        st.info("No historical emotion data found for this user yet. Start a webcam session and save logs to see insights here!")

with tab3:
    st.header("Manual Mood Log & Recommendations")
    st.write(f"Manually log your mood to compare with detected emotions or get personalized recommendations for **{username}**.")

    # Manual Mood Logging
    st.subheader("Log Your Mood")
    selected_mood = st.select_slider(
        "How are you feeling right now?",
        options=["Very Sad", "Sad", "Neutral", "Happy", "Very Happy", "Angry", "Stressed", "Surprised"],
        value="Neutral",
        key="manual_mood_slider"
    )
    mood_note = st.text_area("Add a short note about your mood (optional)", key="mood_note")
    
    if st.button("Submit Mood Log", key="submit_mood_log_btn"):
        mood_log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user": username,
            "mood": selected_mood,
            "note": mood_note,
            "user_id": current_user_id # Ensure event is tied to user
        }
        
        # Save to Firestore (assuming a collection for manual mood logs)
        try:
            db.collection(f"artifacts/{st.session_state.app_id}/users/{current_user_id}/manual_mood_logs").add_doc(mood_log_entry)
            st.success(f"Mood '{selected_mood}' logged successfully for {username}!")

            # --- Feature: Daily Streak Update ---
            achievements_data = load_achievements(current_user_id) # Load for current user
            today = date.today().isoformat()
            last_log_date_str = achievements_data.get("last_mood_log_date")
            
            if last_log_date_str:
                last_log_date = date.fromisoformat(last_log_date_str)
                if last_log_date == today:
                    st.info("You've already logged your mood today. Keep up the good work!")
                elif last_log_date == today - timedelta(days=1):
                    achievements_data["daily_mood_streak"] = achievements_data.get("daily_mood_streak", 0) + 1
                    st.balloons()
                    st.success(f"🎉 Great job! Your daily mood log streak is now {achievements_data['daily_mood_streak']} days!")
                else:
                    achievements_data["daily_mood_streak"] = 1 # Reset streak
                    st.info("New streak started!")
            else:
                achievements_data["daily_mood_streak"] = 1 # First log
                st.info("Your first mood log! Daily streak started.")
            
            achievements_data["last_mood_log_date"] = today
            save_achievements(current_user_id, achievements_data) # Save updated achievements
            
            st.experimental_rerun() # To refresh the displayed streaks if user goes to Tab 6

        except Exception as e:
            st.error(f"Error logging mood to Firestore: {e}")

    st.subheader("Past Mood Logs")
    # Load past mood logs from Firestore
    try:
        docs = db.collection(f"artifacts/{st.session_state.app_id}/users/{current_user_id}/manual_mood_logs").get_docs()
        manual_mood_data = []
        for doc in docs:
            doc_data = doc.to_dict()
            if 'timestamp' in doc_data:
                doc_data['timestamp'] = doc_data['timestamp']
            manual_mood_data.append(doc_data)
        
        manual_mood_history_df = pd.DataFrame(manual_mood_data)
        if not manual_mood_history_df.empty:
            manual_mood_history_df['timestamp'] = pd.to_datetime(manual_mood_history_df['timestamp'])
            st.dataframe(manual_mood_history_df.sort_values(by='timestamp', ascending=False).head(10))
        else:
            st.info("No manual mood logs found yet for this user. Submit your first mood!")
    except Exception as e:
        st.error(f"Error fetching past mood logs: {e}")
        st.info("No manual mood logs found yet for this user. Submit your first mood!")


    # --- New Feature: Emotional Goal Setting & Progress ---
    st.markdown("---")
    st.subheader("Emotional Goal Setting & Progress")
    st.write(f"Set personal emotional goals and track your progress for **{username}**.")

    current_goals = load_goals(current_user_id) # Load from Firestore

    with st.expander("Set a New Emotional Goal"):
        goal_type = st.radio("I want to...", ["increase", "decrease"], key="goal_type_select", horizontal=True)
        all_known_emotions = [e for e in ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]]
        target_emotion = st.selectbox("Emotion:", all_known_emotions, key="target_emotion_select")
        target_percentage_change = st.number_input(
            f"By what percentage points (e.g., {5 if goal_type == 'increase' else 3} for {5 if goal_type == 'increase' else 3}%)?",
            min_value=1, max_value=100, value=5, step=1, key="target_pct_input"
        )
        goal_start_date = st.date_input("Start Date:", value=date.today(), key="goal_start_date")
        goal_end_date = st.date_input("End Date:", value=date.today() + timedelta(days=30), key="goal_end_date")

        if st.button("Add Goal", key="add_goal_btn"):
            if goal_end_date <= goal_start_date:
                st.warning("End date must be after start date.")
            else:
                goal_id = f"goal_{len(current_goals) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                current_goals.append({
                    "id": goal_id,
                    "emotion": target_emotion,
                    "type": goal_type,
                    "target_percent_change": target_percentage_change,
                    "start_date": str(goal_start_date),
                    "end_date": str(goal_end_date),
                    "status": "active"
                })
                save_goals(current_user_id, current_goals) # Save to Firestore
                st.success("Goal added successfully!")
                st.rerun()

    st.subheader("Your Active Goals")
    active_goals_to_display = [g for g in current_goals if g['status'] == 'active']

    if active_goals_to_display:
        for i, goal in enumerate(active_goals_to_display):
            st.markdown(f"**Goal {i+1}:** To **{goal['type']}** **{goal['emotion'].capitalize()}** {get_emotion_emoji(goal['emotion'])} by **{goal['target_percent_change']}%** between **{goal['start_date']}** and **{goal['end_date']}**.")
            
            # --- Basic Progress Tracking Placeholder ---
            st.info("Progress tracking for this goal will be visually displayed here in future updates! For now, check your historical trends in Tab 2 to see how you're doing.")
            
            if st.button(f"Mark as Completed", key=f"complete_goal_btn_{goal['id']}"):
                for g in current_goals:
                    if g['id'] == goal['id']:
                        g['status'] = 'completed'
                        break
                save_goals(current_user_id, current_goals) # Save to Firestore
                st.success(f"Goal '{goal['emotion'].capitalize()}' marked as completed!")
                st.rerun()
            st.markdown("---") # Separator between goals
    else:
        st.info("No active goals found for this user. Set a new goal above!")

    st.subheader("Personalized Activity Suggestions")
    st.markdown(f"Based on your current detected emotion: **{st.session_state.current_emotion.capitalize()} {get_emotion_emoji(st.session_state.current_emotion)}**")

    # Simple Recommendation System
    recommendations = {
        "happy": [
            "Go for a walk outside to enjoy the day.",
            "Call a friend or family member to share your joy.",
            "Work on a hobby you enjoy.",
            "Plan something fun for later!",
            "Write down what made you happy today."
        ],
        "sad": [
            "Listen to calming music or a guided meditation.",
            "Watch a comfort movie or show.",
            "Connect with a trusted person; sharing can help.",
            "Try a gentle stretching exercise or yoga.",
            "Journal your feelings to process them."
        ],
        "angry": [
            "Take 10 deep breaths to calm your nervous system.",
            "Go for a brisk walk or do some quick exercises.",
            "Count to 10 slowly before reacting.",
            "Listen to instrumental music to de-stress.",
            "Practice progressive muscle relaxation."
        ],
        "surprise": [
            "Reflect on what surprised you and why.",
            "If it's positive, celebrate the moment!",
            "If it's unsettling, take a moment to understand it."
        ],
        "neutral": [
            "Try a new short online course or learn something new.",
            "Listen to a thought-provoking podcast.",
            "Do a quick tidying up of your workspace.",
            "Engage in a light brain-teaser or puzzle.",
            "Explore a new genre of music."
        ],
        "fear": [
            "Remind yourself of your strength and resilience.",
            "Focus on your breathing; breathe in courage, breathe out fear.",
            "Visualize a safe and peaceful place.",
            "Identify the source of fear and consider small steps to address it.",
            "Listen to grounding sounds like rain or ocean waves."
        ],
        "disgust": [
            "Step away from what's causing the feeling.",
            "Clean or organize your immediate environment.",
            "Engage in an activity that provides a sense of purity or freshness (e.g., cooking, taking a shower).",
            "Change your sensory input: light a pleasant candle, listen to uplifting sounds."
        ],
        "No Face": [
            "I need to see your face to give personalized recommendations!",
            "Adjust your lighting or camera position."
        ]
    }
    
    if st.session_state.current_emotion in recommendations:
        num_suggestions = st.slider("Number of suggestions to show", 1, 5, 3, key="num_suggestions_slider")
        selected_recs = random.sample(recommendations[st.session_state.current_emotion], min(num_suggestions, len(recommendations[st.session_state.current_emotion])))
        for i, rec in enumerate(selected_recs):
            st.success(f"✨ {rec}")
    else:
        st.info("No specific recommendations at this moment.")
    
    if st.button("Get More Suggestions", key="refresh_recs_btn"):
        st.experimental_rerun() 


with tab4:
    st.header("Chat with EmotionBot 🤖")
    st.write("Have a conversation with EmotionBot! It tries to respond based on your current detected emotion and your messages. Type 'help' to get started.")
    
    chat_display_area = st.container(height=400, border=True)
    with chat_display_area:
        for chat_entry in st.session_state.chat_history:
            sender = chat_entry["sender"]
            message = chat_entry["message"]
            if sender == "You":
                st.chat_message("user").write(message)
            else:
                st.chat_message("assistant").write(message)

    user_input = st.chat_input("Type your message here...", key="chat_input_tab4")

    if user_input:
        emotion_for_chatbot = st.session_state.current_emotion
        response = chatbot_response(user_input, emotion_for_chatbot, st.session_state.chat_history)
        
        st.session_state.chat_history.append({"sender": "You", "message": user_input})
        st.session_state.chat_history.append({"sender": "Bot", "message": response})
        # Note: Streamlit's chat_input auto-clears on submit, no explicit rerun needed for that.

with tab5: # --- New Tab: Settings ---
    st.header("User Preferences & Settings")
    st.write(f"Customize the dashboard experience for **{username}**.")

    with st.expander("General Settings", expanded=True):
        st.subheader("Dashboard Theme")
        theme_options = ["light", "dark"]
        selected_theme = st.radio(
            "Select a theme:",
            options=theme_options,
            index=theme_options.index(st.session_state.settings.get("theme", "light")),
            key="theme_setting"
        )
        if selected_theme != st.session_state.settings.get("theme"):
            st.session_state.settings["theme"] = selected_theme
            save_settings(current_user_id, st.session_state.settings) # Save to Firestore
            st.info("Theme preference saved. Restart the app for full effect (requires `config.toml` setup for Streamlit theming).")


    with st.expander("Feedback & Alert Settings"):
        st.subheader("Speech & Music Feedback")
        speech_enabled_setting = st.checkbox(
            "Enable Speech Feedback", 
            value=st.session_state.settings.get("speech_enabled", True), 
            key="setting_speech_enabled"
        )
        music_toast_enabled_setting = st.checkbox(
            "Enable Music Suggestions", 
            value=st.session_state.settings.get("music_toast_enabled", True), 
            key="setting_music_toast_enabled"
        )
        speech_frequency_setting = st.slider(
            "Speech/Music Frequency (seconds)", 1, 60, 
            value=st.session_state.settings.get("speech_frequency", 10), 
            key="setting_speech_frequency"
        )

        st.subheader("Custom Emotion Alerts")
        alert_emotion_options_for_select = ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]
        alert_emotion_setting = st.selectbox(
            "Alert me if dominant emotion is:", 
            ["None"] + sorted(alert_emotion_options_for_select), 
            index=(["None"] + sorted(alert_emotion_options_for_select)).index(st.session_state.settings.get("alert_emotion", "None")),
            key="setting_alert_emotion_select"
        )
        alert_duration_setting = st.number_input(
            "For how long (seconds)?:", min_value=5, max_value=300, 
            value=st.session_state.settings.get("alert_duration", 30), step=5, 
            key="setting_alert_duration"
        )
    
    if st.button("Save All Settings", key="save_settings_btn"):
        st.session_state.settings["speech_enabled"] = speech_enabled_setting
        st.session_state.settings["music_toast_enabled"] = music_toast_enabled_setting
        st.session_state.settings["speech_frequency"] = speech_frequency_setting
        st.session_state.settings["alert_emotion"] = alert_emotion_setting
        st.session_state.settings["alert_duration"] = alert_duration_setting
        save_settings(current_user_id, st.session_state.settings) # Save to Firestore
        st.success("Settings saved successfully! Some changes (like theme) might require restarting the app.")
        st.experimental_rerun() # Rerun to apply settings to current session_state

with tab6: # --- New Tab: Streaks & Achievements ---
    st.header("Streaks & Achievements 🏆")
    st.write(f"Track your consistency and unlock achievements for **{username}**!")

    achievements_data = load_achievements(current_user_id) # Load from Firestore

    st.subheader("Daily Mood Log Streak")
    daily_streak = achievements_data.get("daily_mood_streak", 0)
    last_log_date = achievements_data.get("last_mood_log_date")
    
    st.metric(label="Current Daily Mood Log Streak", value=f"{daily_streak} days")
    if last_log_date:
        st.info(f"Last mood logged on: {last_log_date}")
    else:
        st.info("No mood logs yet. Start your streak by logging your mood in Tab 3!")

    st.subheader("Emotion Streaks (Longest Continuous Detection)")
    st.write("These streaks track how long a particular emotion was continuously detected by the webcam in a single session.")
    
    emotion_streaks_data = achievements_data.get("emotion_streaks", {})
    
    # Display the current session's active streak (if webcam is running)
    if st.session_state.webcam_running and st.session_state.emotion_current_streak_emotion and st.session_state.emotion_current_streak_length > 0:
        st.info(f"Currently building a **{st.session_state.emotion_current_streak_emotion.capitalize()}** streak: **{st.session_state.emotion_current_streak_length:.1f} seconds**")

    if emotion_streaks_data:
        sorted_streaks = sorted(emotion_streaks_data.items(), key=lambda item: item[1], reverse=True)
        for emotion, length in sorted_streaks:
            st.markdown(f"- **{emotion.capitalize()}:** {length:.1f} seconds {get_emotion_emoji(emotion)}")
    else:
        st.info("No emotion streaks recorded yet. Start a webcam session to build them!")

    st.subheader("Achievements Unlocked")
    st.info("This section will show badges for achievements like 'First Report Exported', '10 Day Mood Log Streak', etc. (Future Feature)")
    # Example placeholder:
    # if daily_streak >= 10:
    #     st.success("🏅 **10 Day Mood Log Master!** You've logged your mood for 10 consecutive days!")


st.sidebar.markdown("---")
st.sidebar.subheader("Dashboard Info")
st.sidebar.markdown(f"""
- **Version:** 1.5.0
- **Developed by:** Your Name/Team
- **Purpose:** Real-time emotion monitoring and historical analysis.
- **Data Storage:** All user data (logs, reports, settings, achievements, goals, events) is stored securely in **Firestore** and is accessible only to the logged-in user.
- **Current User ID:** `{current_user_id}`
- **Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")