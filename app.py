import streamlit as st
import requests
import tempfile
import uuid
from datetime import datetime
import pandas as pd
from typing import Dict, Any
from utils import INDIC_CONFORMER_LANGUAGES, WHISPER_LANGUAGES
import hashlib
import os
import time

def hash_password(password: str) -> str:
    """Simple password hashing (use proper hashing in production)"""
    return hashlib.sha256(password.encode()).hexdigest()

# ===========================
# Configuration
# ===========================
API_BASE_URL = "http://localhost:9001"  # Update with your API URL
API_ENDPOINT = "/speech-to-text-service"
USERS = {
    "demo": hash_password("demo"),
}
# ===========================
# Session State Management
# ===========================
def init_session_state():
    """Initialize session state variables"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'request_history' not in st.session_state:
        st.session_state.request_history = []
    if 'current_request_id' not in st.session_state:
        st.session_state.current_request_id = None
    if 'last_response' not in st.session_state:
        st.session_state.last_response = None

# ===========================
# Authentication Functions
# ===========================

def authenticate_user(username: str, password: str) -> bool:
    """
    Authenticate user - in production, this would check against a database
    For demo purposes, using simple credentials
    """
    # Demo credentials (replace with actual authentication)
    
    hashed_pwd = hash_password(password)
    return USERS.get(username) == hashed_pwd

def generate_session_id(user_id: str) -> str:
    """Generate a unique session ID based on user ID and timestamp"""
    timestamp = str(int(time.time()))
    session_string = f"{user_id}_{timestamp}"
    return hashlib.md5(session_string.encode()).hexdigest()[:16]

# ===========================
# API Functions
# ===========================
def generate_request_id(user_id: str) -> str:
    """Generate unique request ID with user context"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{user_id}_{timestamp}_{unique_id}"

def call_stt_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call the speech-to-text API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}{API_ENDPOINT}",
            json=payload,
            timeout=300  # 5 minutes timeout for long audio processing
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ===========================
# UI Components
# ===========================
def render_login_page():
    """Render login page"""
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: auto;
        padding: 2rem;
        background: #f0f2f6;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("## 🔐 Login")
        st.markdown("---")
        
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
                
        if st.button("🚀 Login", width='stretch', type="primary"):
            if username and password:
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.user_id = username
                    st.session_state.session_id = generate_session_id(username)
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
            else:
                st.warning("⚠️ Please enter both username and password")

        # Demo credentials info
        with st.expander("📋 Demo Credentials"):
            st.markdown("""
            **Test Accounts:**
            - Username: `demo` | Password: `demo`
            """)

def render_sidebar():
    """Render sidebar with user info and settings"""
    with st.sidebar:
        st.markdown("## 👤 User Information")
        st.markdown(f"**User ID:** `{st.session_state.user_id}`")
        st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
        
        st.markdown("---")
        
        # Request history
        st.markdown("## 📊 Request History")
        if st.session_state.request_history:
            for idx, req in enumerate(reversed(st.session_state.request_history[-5:])):
                with st.expander(f"Request {len(st.session_state.request_history) - idx}"):
                    st.text(f"ID: {req['req_id']}")
                    st.text(f"Time: {req['timestamp']}")
                    st.text(f"Status: {req['status']}")
        else:
            st.info("No requests yet")
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", width='stretch'):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def render_main_app():
    """Render main application"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white;">🎙️ Speech-to-Text Service</h1>
        <p style="color: white;">Professional transcription pipeline with advanced features</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    tabs = st.tabs(["📤 New Request", "📊 Results", "📊 Statistics"])
    
    with tabs[0]:
        render_request_form()
    
    with tabs[1]:
        render_results()
    
    with tabs[2]:
        render_statistics()

def render_request_form():
    """Render the request form for STT service"""
    st.markdown("### Configure Transcription Request")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎵 Audio Input")
        audio_source = st.radio(
            "Select audio source:",
            ["Upload File", "URL"],
            horizontal=True
        )
        
        audio_url = ""
        audio_path = ""
        
        if audio_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg']
            )
            if uploaded_file:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                # Save file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    audio_path = tmp_file.name  # store path for later use

                st.info(f"File saved temporarily at: {audio_path}")

        elif audio_source == "URL":
            audio_url = st.text_input(
                "Audio URL",
                placeholder="https://example.com/audio.mp3"
            )
        
        st.markdown("#### 🛠️ Service Configuration")
        stt_service = st.selectbox(
            "STT Service",
            ["whisper", "faster-whisper", "indic-conformer"],
            help="Select the speech-to-text service provider"
        )
        
        language = st.selectbox(
            "Language",
            INDIC_CONFORMER_LANGUAGES if stt_service == "indic-conformer" else WHISPER_LANGUAGES,
            help="Select the audio language"
        )
        
        output_format = st.selectbox(
            "Output Format",
            ["srt", "txt"],
            help="Select the output format for transcription"
        )
    
    with col2:
        st.markdown("#### 🔧 Processing Options")
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            wlt = st.checkbox("WLT Processing", value=True, help="Word-level timestamps")
            uvr = st.checkbox("UVR Processing", value=True, help="Vocal removal")
            maintain_gaps = st.checkbox("Maintain Gaps", value=True, help="Preserve silence gaps")
        
        with col_opt2:
            vad = st.checkbox("VAD Processing", value=True, help="Voice activity detection")
            diarize = st.checkbox("Speaker Diarization", value=False, help="Identify different speakers")
        
        st.markdown("#### 📝 Pattern Configuration")
        with st.expander("Advanced Pattern Settings"):
            start_pattern = st.text_input("Start Pattern", value="[{", help="Pattern to identify start")
            end_pattern = st.text_input("End Pattern", value="}]: ", help="Pattern to identify end")
    
    # Submit button
    st.markdown("---")
    col_submit1, col_submit2, col_submit3 = st.columns([1, 2, 1])
    
    with col_submit2:
        if st.button("🚀 Start Transcription", width='stretch', type="primary"):
            if audio_url or audio_path or audio_source == "Upload File":
                # Generate request ID
                req_id = generate_request_id(st.session_state.user_id)
                st.session_state.current_request_id = req_id
                
                # Prepare payload
                payload = {
                    "audio_url": audio_url,
                    "audio_path": audio_path,
                    "stt_service": stt_service,
                    "start_pattern": start_pattern,
                    "end_pattern": end_pattern,
                    "language": language,
                    "wlt": wlt,
                    "uvr": uvr,
                    "vad": vad,
                    "diarize": diarize,
                    "output_format": output_format,
                    "req_id": req_id,
                    "maintain_gaps": maintain_gaps
                }
                
                # Show processing message
                with st.spinner("🔄 Processing your request... This may take a few minutes."):
                    response = call_stt_api(payload)
                
                # Store response
                st.session_state.last_response = response
                
                # Add to history
                history_entry = {
                    "req_id": req_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": response.get("status", "unknown"),
                    "response": response
                }
                st.session_state.request_history.append(history_entry)
                
                # Show result
                if response.get("status") == "success":
                    st.success("✅ Transcription completed successfully!")
                    st.balloons()
                else:
                    st.error(f"❌ Error: {response.get('message', 'Unknown error')}")
            else:
                st.warning("⚠️ Please provide audio input (URL, file path, or upload)")

def render_results():
    """Render results section"""
    st.markdown("### 📊 Transcription Results")
    
    if st.session_state.last_response:
        response = st.session_state.last_response
        
        if response.get("status") == "success":
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Request ID", response.get("req_id", "N/A"))
            with col2:
                st.metric("Status", "✅ Success")
            with col3:
                segments = response.get("segments", [])
                st.metric("Segments", len(segments) if segments else "N/A")
            
            # Transcript
            st.markdown("#### 📝 Transcript")
            transcript = response.get("transcript", "")
            if transcript:
                st.text_area(
                    "Full Transcript",
                    transcript,
                    height=200,
                    disabled=True
                )
                
                # Download button
                st.download_button(
                    label="📥 Download Transcript",
                    data=transcript,
                    file_name=f"transcript_{st.session_state.current_request_id}.txt",
                    mime="text/plain"
                )
            
            # # Segments
            # if segments:
            #     st.markdown("#### 🎯 Segments")
            #     with st.expander("View Segments Details"):
            #         df_segments = pd.DataFrame(segments)
            #         st.dataframe(df_segments, width='stretch')
            
            # Original speakers
            speakers = response.get("original_speakers", [])
            if speakers:
                st.markdown("#### 👥 Speakers")
                st.write(speakers)
            
            # Output file path
            output_file = response.get("output_file_path", "")
            if output_file:
                st.markdown("#### 📁 Output File")
                st.info(f"File saved at: `{output_file}`")
        
        else:
            st.error(f"Last request failed: {response.get('message', 'Unknown error')}")
    else:
        st.info("No results yet. Please make a transcription request.")

def render_statistics():
    """Render statistics page"""
    st.markdown("### 📊 Statistics")
    
    if st.session_state.request_history:
        total_requests = len(st.session_state.request_history)
        successful_requests = sum(1 for req in st.session_state.request_history if req['status'] == 'success')
        failed_requests = total_requests - successful_requests
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Requests", total_requests)
        with col_stat2:
            st.metric("Successful", successful_requests)
        with col_stat3:
            st.metric("Failed", failed_requests)
        
        # Request history table
        st.markdown("#### 📜 Full Request History")
        history_data = []
        for req in st.session_state.request_history:
            history_data.append({
                "Request ID": req['req_id'],
                "Timestamp": req['timestamp'],
                "Status": "✅" if req['status'] == 'success' else "❌"
            })
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, width='stretch')
        
        if st.button("🗑️ Clear History"):
            st.session_state.request_history = []
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No statistics available yet.")

# ===========================
# Main Application
# ===========================
def main():
    st.set_page_config(
        page_title="Speech-to-Text Service",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Check authentication
    if not st.session_state.authenticated:
        render_login_page()
    else:
        render_main_app()

if __name__ == "__main__":
    main()