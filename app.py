import streamlit as st
import os
import queue
import threading
import time
import av
import numpy as np
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from openai import OpenAI
from utils import get_text_from_upload
import io

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="Live Interview Copilot", layout="wide", initial_sidebar_state="expanded")



# --- Global State (Singleton-like) ---
if "audio_buffer" not in st.session_state:
    st.session_state["audio_buffer"] = queue.Queue()
if "text_buffer" not in st.session_state:
    st.session_state["text_buffer"] = queue.Queue()
if "last_transcript" not in st.session_state:
    st.session_state["last_transcript"] = ""

# --- Classes & Functions ---

class AudioProcessor:
    def __init__(self):
        self.frame_buffer = []
        self.sample_rate = 48000  # Default for WebRTC
        self.last_process_time = time.time()
        self.chunk_duration = 5.0 # Process every 5 seconds

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert to numpy and store
        audio_data = frame.to_ndarray()
        self.frame_buffer.append(audio_data)
        
        # Check if enough time has passed to process a chunk
        current_time = time.time()
        if current_time - self.last_process_time > self.chunk_duration:
            if self.frame_buffer:
                # Combine frames
                combined_audio = np.concatenate(self.frame_buffer, axis=1)
                
                # We need to send this to the main thread or process it here.
                # Processing here might block audio (bad).
                # So we push a copy to a global queue.
                # Note: We can't access st.session_state directly in this thread safely in all versions,
                # but in newer Streamlit/webrtc it might work. 
                # Better to use a dedicated queue passed in context, but for simplicity:
                
                # Ideally, we would use a callback. 
                # For this MVP, we'll try to rely on the fact that we can access global vars if careful.
                # But to be safe, we will just clear buffer and continue.
                
                # Actually, let's use the queue we defined in session_state?
                # No, session_state is thread-local.
                # We need a true global or resource.
                pass 
                
            self.frame_buffer = []
            self.last_process_time = current_time
            
        return frame

# To share data between the WebRTC thread and the main thread, we use a global queue via st.cache_resource
@st.cache_resource
def get_audio_queue():
    return queue.Queue()

@st.cache_resource
def get_transcript_queue():
    return queue.Queue()

audio_queue = get_audio_queue()
transcript_queue = get_transcript_queue()

class AudioProcessorV2:
    def __init__(self):
        self.frame_buffer = []
        self.sample_rate = 48000
        self.chunk_duration = 3.0 # 3 seconds chunks
        self.last_time = time.time()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            # frame.format.name is usually 's16' (16-bit PCM) or 'fltp' (float planar)
            # We convert to ndarray. 
            # frame.to_ndarray() returns (channels, samples) or (samples, channels) depending on layout
            data = frame.to_ndarray()
            self.frame_buffer.append(data)
            
            now = time.time()
            if now - self.last_time > self.chunk_duration:
                if len(self.frame_buffer) > 0:
                    # Concatenate
                    # Note: We need to be careful with dimensions.
                    # Usually (2, 480) for stereo 10ms frame at 48k? 
                    # Let's assume standard concatenation works along the time axis (1).
                    combined = np.concatenate(self.frame_buffer, axis=1)
                    
                    # Put into queue
                    audio_queue.put((combined, self.sample_rate))
                    
                    self.frame_buffer = []
                    self.last_time = now
        except Exception as e:
            print(f"Error in recv: {e}")
            
        return frame

import wave

def process_audio_chunk(audio_data, sample_rate, client):
    """
    Convert numpy array to WAV and transcribe.
    audio_data: (channels, samples)
    """
    try:
        # Check dimensions
        if audio_data.ndim == 1:
            channels = 1
            samples = audio_data.shape[0]
        else:
            channels, samples = audio_data.shape
            # If shape is (channels, samples), we might need to transpose for some operations, 
            # but for wave module, we need interleaved bytes usually? 
            # Or separate channels?
            # Standard WAV is interleaved.
            # If we have (2, N), we need to interleave.
            # audio_data.T is (N, 2). Flattening it gives interleaved [L, R, L, R...]
            audio_data = audio_data.T

        # Convert to int16 if needed
        if audio_data.dtype == np.float32:
            audio_data = (audio_data * 32767).astype(np.int16)
        elif audio_data.dtype != np.int16:
             # Try to cast
             audio_data = audio_data.astype(np.int16)

        # Write to WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2) # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        buffer.name = "audio.wav"
        
        # Transcribe
        if client:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=buffer,
                language="en" 
            )
            return transcript.text
    except Exception as e:
        print(f"Processing error: {e}")
        return None
    return ""

def generate_ai_response(transcript_text, context_text, client):
    if not transcript_text.strip():
        return None
        
    try:
        messages = [
            {"role": "system", "content": """You are the user, a job candidate in a live interview. 
Your goal is to provide a concise, natural, and impactful answer to the interviewer's question. 
Speak in the first person ('I have experience with...', 'I believe...'). 
Do not explain what you are doing. Do not say 'Here is an answer'. 
Just give the direct answer as if you are speaking it. 
Keep it conversational but professional.

CRITICAL INSTRUCTIONS:
1. CHECK THE SCRIPT: If the interviewer's question matches or is very similar to any question/topic in the provided SCRIPT, YOU MUST ADAPT THE ANSWER FROM THE SCRIPT. The script is your primary source of truth for those specific questions.
2. If the question is not in the script, use the RESUME and JOB DESCRIPTION to construct a relevant answer.
3. Always maintain the persona of the candidate. Never break character."""},
            {"role": "user", "content": f"My Resume, JD & Script Context:\n{context_text}\n\nInterviewer Question/Transcript:\n{transcript_text}"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

# --- UI ---

st.title("🎤 Live Interview AI Copilot")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        client = OpenAI(api_key=api_key)
    else:
        st.warning("Enter OpenAI API Key")
        client = None
    st.subheader("Context Materials")
    resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx", "txt"])
    job_desc = st.text_area("Paste Job Description Here", height=150, placeholder="Copy and paste the job description...")
    
    st.markdown("### Interview Script")
    script_file = st.file_uploader("Upload Script (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
    script_text = st.text_area("Or Paste Script Here", height=100, placeholder="Paste any known questions/script...")

    if st.button("Load Context"):
        r_text = get_text_from_upload(resume_file)
        s_text_file = get_text_from_upload(script_file)
        
        # Combine uploaded script and pasted script
        full_script = f"{s_text_file}\n\n{script_text}".strip()
        
        st.session_state['context_text'] = f"RESUME:\n{r_text}\n\nJD:\n{job_desc}\n\nSCRIPT:\n{full_script}"
        st.success("Context Loaded")

    st.markdown("---")
    hud_position = st.selectbox("HUD Position", ["Top-Right", "Top-Left", "Bottom-Right", "Bottom-Left", "Top-Center"], index=0)

    # Map selection to CSS coordinates
    hud_css = ""
    if hud_position == "Top-Right":
        hud_css = "top: 20px; right: 20px;"
    elif hud_position == "Top-Left":
        hud_css = "top: 20px; left: 20px;"
    elif hud_position == "Bottom-Right":
        hud_css = "bottom: 20px; right: 20px;"
    elif hud_position == "Bottom-Left":
        hud_css = "bottom: 20px; left: 20px;"
    elif hud_position == "Top-Center":
        hud_css = "top: 20px; left: 50%; transform: translateX(-50%);"

    if st.button("Clear Transcript"):
        st.session_state["last_transcript"] = ""
        st.rerun()

# --- Custom CSS for HUD/Overlay Mode ---
st.markdown(f"""
<style>
    /* Compact HUD Style for the Answer Box */
    .floating-answer-box {{
        position: fixed;
        {hud_css}
        width: 400px;
        background-color: rgba(0, 0, 0, 0.85);
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        z-index: 9999;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #444;
        font-family: sans-serif;
    }}
    .floating-answer-box h4 {{
        margin-top: 0;
        color: #00ff00;
        font-size: 16px;
    }}
    .floating-answer-box p {{
        font-size: 18px;
        line-height: 1.4;
    }}
    
    /* Make the main content cleaner */
    .block-container {{
        padding-top: 2rem;
    }}
</style>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Audio Stream")
    ctx = webrtc_streamer(
        key="interview-ai",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessorV2,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": False, "audio": True}
    )

with col2:
    st.subheader("Live Assistant")
    transcript_placeholder = st.empty()
    suggestion_placeholder = st.empty()

# Main Loop for processing
if ctx.state.playing and client:
    # We check the queue periodically
    while ctx.state.playing:
        try:
            # Non-blocking get
            audio_data, rate = audio_queue.get_nowait()
            
            # Process
            with st.spinner("Transcribing..."):
                text = process_audio_chunk(audio_data, rate, client)
            
            if text:
                st.session_state["last_transcript"] += f"\nUser/Interviewer: {text}"
                transcript_placeholder.markdown(st.session_state["last_transcript"])
                
                # Generate Answer
                if "context_text" in st.session_state:
                    with st.spinner("Thinking..."):
                        answer = generate_ai_response(text, st.session_state['context_text'], client)
                        
                        # Display in Floating HUD
                        suggestion_placeholder.markdown(
                            f"""
                            <div class="floating-answer-box">
                                <h4>AI Suggestion</h4>
                                <p>{answer}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            st.error(f"Loop Error: {e}")
            break
