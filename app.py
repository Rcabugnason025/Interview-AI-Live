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
import sounddevice as sd

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

# --- Local Audio Capture (System Audio) ---
# This runs in a background thread when enabled
def audio_callback(indata, frames, time, status):
    """Callback for sounddevice input stream."""
    if status:
        print(status)
    # indata is (frames, channels) float32
    # We need to buffer this.
    # To keep it compatible with our existing queue which expects (channels, samples) or similar?
    # Our existing processor uses (2, N) usually.
    # sounddevice gives (N, channels). We should transpose it.
    audio_queue.put((indata.T, 16000)) # Using 16k for local capture usually good for Whisper

@st.cache_resource
class SystemAudioRecorder:
    def __init__(self):
        self.stream = None
        self.is_running = False

    def start(self):
        if self.is_running:
            return
            
        # Try to find a loopback device or default input
        # For WASAPI loopback, we usually need to find a device with "loopback"
        # However, selecting default input might just be the mic.
        # Let's try to find a device that looks like a loopback or "Stereo Mix"
        device_id = None
        try:
            devices = sd.query_devices()
            # On Windows WASAPI, we can use loopback=True on the default output device
            # This is supported in recent sounddevice versions
            pass
        except:
            pass
            
        try:
            # Attempt WASAPI loopback on default output
            # This requires 'sounddevice' to be installed with WASAPI support (standard on Windows)
            # loopback=True is the key
            self.stream = sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=16000,
                loopback=True # This captures system audio!
            )
            self.stream.start()
            self.is_running = True
            print("System Audio Recording Started")
        except Exception as e:
            st.error(f"Could not start System Audio Capture: {e}. Falling back to default microphone.")
            try:
                self.stream = sd.InputStream(
                    callback=audio_callback,
                    channels=1,
                    samplerate=16000
                )
                self.stream.start()
                self.is_running = True
            except Exception as e2:
                st.error(f"Failed to start microphone: {e2}")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_running = False
        print("System Audio Recording Stopped")

system_recorder = SystemAudioRecorder()


import wave

def process_audio_chunk(audio_data, sample_rate, client):
    """
    Convert numpy array to WAV and transcribe.
    audio_data: (channels, samples) or (samples, channels)
    """
    try:
        # Check dimensions
        if audio_data.ndim == 1:
            channels = 1
            samples = audio_data.shape[0]
        else:
            # If shape is (channels, samples), we might need to transpose for some operations, 
            # but for wave module, we need interleaved bytes usually? 
            # Or separate channels?
            # Standard WAV is interleaved.
            # If we have (2, N), we need to interleave.
            # audio_data.T is (N, 2). Flattening it gives interleaved [L, R, L, R...]
            
            # sounddevice gives (N, channels). Transposed to (channels, N).
            # If it is (channels, N), we transpose back to (N, channels) for wave writing usually?
            # Wave writeframes expects bytes.
            
            # Let's handle both. We want (N, channels) for flattening.
            if audio_data.shape[0] < audio_data.shape[1]:
                # It is likely (channels, samples) -> Transpose to (samples, channels)
                audio_data = audio_data.T
            
            samples, channels = audio_data.shape

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
    
    st.subheader("Audio Settings")
    audio_source = st.radio("Audio Source", ["Browser Microphone (WebRTC)", "System Audio (Windows Loopback)"], index=1)
    st.caption("Use 'System Audio' to capture the Interviewer's voice from your speakers.")

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
    
    if audio_source == "Browser Microphone (WebRTC)":
        # Stop local recorder if running
        if system_recorder.is_running:
            system_recorder.stop()
            
        ctx = webrtc_streamer(
            key="interview-ai",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=AudioProcessorV2,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": False, "audio": True}
        )
        is_playing = ctx.state.playing
    else:
        # System Audio Mode
        st.info("Listening to System Audio (what you hear). Ensure your speakers are on.")
        if st.button("Start Listening"):
            system_recorder.start()
        
        if st.button("Stop Listening"):
            system_recorder.stop()
            
        if system_recorder.is_running:
            st.success("🔴 Recording System Audio...")
            is_playing = True
        else:
            is_playing = False

with col2:
    st.subheader("Live Assistant")
    transcript_placeholder = st.empty()
    suggestion_placeholder = st.empty()

# Main Loop for processing
if is_playing and client:
    # We check the queue periodically
    while is_playing:
        try:
            # Non-blocking get
            audio_data, rate = audio_queue.get_nowait()
            
            # Process
            text = process_audio_chunk(audio_data, rate, client)
            
            if text:
                # Update transcript
                current_transcript = st.session_state["last_transcript"] + " " + text
                st.session_state["last_transcript"] = current_transcript[-2000:] # Keep last 2000 chars
                
                # Show transcript
                transcript_placeholder.markdown(f"**Transcript:**\n\n{st.session_state['last_transcript']}")
                
                # Generate AI Response
                context = st.session_state.get('context_text', "No context loaded.")
                ai_answer = generate_ai_response(text, context, client)
                
                if ai_answer:
                    # Show in floating HUD
                    suggestion_placeholder.markdown(f"""
                    <div class="floating-answer-box">
                        <h4>AI Suggested Answer:</h4>
                        <p>{ai_answer}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in main loop: {e}")
            break
