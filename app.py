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

import pyaudiowpatch as pyaudio
import scipy.signal

# Initialize PyAudio
p = pyaudio.PyAudio()

@st.cache_resource
class SystemAudioRecorder:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        # self.queue = queue.Queue() # WRONG: This creates a local queue disconnected from the main app
        self.queue = audio_queue # CORRECT: Use the global queue the main app listens to
        self.is_running = False
        self.device_index = None
        self.native_rate = 44100
        self.native_channels = 2
        self.thread = None

    def set_device(self, index):
        self.device_index = index
        # Get device info to set native rate
        try:
            dev_info = self.pa.get_device_info_by_index(index)
            self.native_rate = int(dev_info.get('defaultSampleRate', 44100))
            self.native_channels = int(dev_info.get('maxInputChannels', 2))
            print(f"Set Device: Index={index}, Rate={self.native_rate}, Channels={self.native_channels}")
        except Exception as e:
            print(f"Error getting device info for index {index}: {e}")
            self.native_rate = 44100 # Fallback

    def get_loopback_devices(self):
        """Returns a list of available loopback devices."""
        devices = []
        try:
            if hasattr(self.pa, 'get_loopback_device_info_generator'):
                for loopback in self.pa.get_loopback_device_info_generator():
                    devices.append({
                        "index": loopback["index"],
                        "name": loopback["name"],
                        "rate": int(loopback.get("defaultSampleRate", 44100))
                    })
        except Exception as e:
            print(f"Error listing devices: {e}")
        return devices

    def auto_scan_and_start(self, start_immediately=True):
        """
        Scans all loopback devices, tries to open a stream, and checks for audio signal.
        If signal > threshold, selects that device and starts.
        """
        print("Starting Deep Scan...")
        st.toast("🔍 Scanning for active audio...", icon="🎧")
        
        devices = self.get_loopback_devices()
        best_device = None
        max_rms = -1.0 
        
        # Temporary PyAudio instance for testing
        temp_pa = pyaudio.PyAudio()
        
        valid_devices = []

        for dev in devices:
            try:
                print(f"Testing: {dev['name']} (Index {dev['index']}) Rate: {dev['rate']}")
                
                # Try to open stream at NATIVE rate
                stream = temp_pa.open(
                    format=pyaudio.paFloat32,
                    channels=2, # Assume stereo
                    rate=dev['rate'], # USE NATIVE RATE
                    input=True,
                    input_device_index=dev['index'],
                    frames_per_buffer=1024
                )
                
                # Read a few chunks
                rms_values = []
                for _ in range(10): # Test for ~0.2 seconds
                    data = stream.read(1024, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    rms = np.sqrt(np.mean(audio_data**2))
                    rms_values.append(rms)
                
                avg_rms = sum(rms_values) / len(rms_values)
                print(f"  > RMS: {avg_rms:.6f}")
                
                valid_devices.append(dev)

                if avg_rms > max_rms:
                    max_rms = avg_rms
                    best_device = dev
                
                stream.stop_stream()
                stream.close()
                
            except Exception as e:
                print(f"  > Failed: {e}")
        
        temp_pa.terminate()
        
        # Fallback: If no "loud" device found, but we found valid ones, pick the first valid one
        if best_device is None and valid_devices:
             best_device = valid_devices[0]
             print(f"Fallback: Selected first valid device: {best_device['name']}")

        if best_device:
             print(f"Selected Best Device: {best_device['name']} (RMS: {max_rms if max_rms > 0 else 0:.6f})")
             self.set_device(best_device['index'])
             
             if start_immediately:
                 self.start()
                 
             return best_device['name']
        
        st.error("⚠️ No audio devices found! Please play some sound (YouTube, Music) and try again.")
        return None

    def start(self):
        if self.is_running:
            return

        # Auto-select device if none selected
        if self.device_index is None:
            print("No device selected. Attempting to auto-select default loopback device...")
            found_name = self.auto_scan_and_start(start_immediately=False)
            if found_name:
                print(f"Auto-selected device: {found_name}")
            else:
                print("No loopback device found during auto-selection.")
                st.error("❌ No System Audio device found. Please select one in the Sidebar.")
                return

        if self.device_index is None:
            print("No device selected for recording (Final Check).")
            return

        try:
            print(f"Starting Recording on Device Index: {self.device_index}")
            
            # Start Background Thread
            self.is_running = True
            self.thread = threading.Thread(target=self._record_loop, daemon=True)
            self.thread.start()
            
            print("Audio Thread Started")
            st.toast("✅ Audio Engine Started (Threaded Mode)", icon="🚀")
            
        except Exception as e:
            print(f"Error starting stream: {e}")
            self.is_running = False

    def _record_loop(self):
        """Blocking read loop in a separate thread (More robust than callbacks)"""
        print(f"THREAD START: Device={self.device_index}, Rate={self.native_rate}")
        try:
            stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self.native_channels,
                rate=self.native_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=1024
            )
            
            print("Stream Opened Successfully in Thread")
            
            # Flush initial buffer
            try:
                stream.read(4096, exception_on_overflow=False)
            except:
                pass

            while self.is_running:
                try:
                    # Blocking Read
                    data = stream.read(1024, exception_on_overflow=False)
                    
                    # Process Data
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    
                    # Mix to Mono if needed
                    if self.native_channels > 1:
                        audio_data = audio_data.reshape(-1, self.native_channels)
                        audio_mono = audio_data.mean(axis=1)
                    else:
                        audio_mono = audio_data

                    # Resample if needed
                    target_rate = 16000
                    if self.native_rate != target_rate:
                        num_samples = int(len(audio_mono) * target_rate / self.native_rate)
                        audio_resampled = scipy.signal.resample(audio_mono, num_samples)
                    else:
                        audio_resampled = audio_mono

                    # Normalize & Boost
                    audio_float = audio_resampled.reshape(-1, 1).astype(np.float32)
                    audio_float = audio_float * 5.0 # Boost
                    np.clip(audio_float, -1.0, 1.0, out=audio_float)
                    
                    # Put in GLOBAL Queue
                    try:
                        audio_queue.put((audio_float, 16000), block=False)
                    except queue.Full:
                        pass # Drop packet if queue is full to avoid hanging the thread
                    
                    # Debug print occasionally
                    if np.random.rand() < 0.005:
                         print("Thread Heartbeat: Chunk Enqueued")
                    
                except OSError as e:
                    if e.errno == -9981: # Input overflowed
                         print("Overflow ignored")
                         continue
                    print(f"OSError in Record Loop: {e}")
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error in Record Loop: {e}")
                    time.sleep(0.1)
            
            stream.stop_stream()
            stream.close()
            print("Stream Closed in Thread")
            
        except Exception as e:
             print(f"CRITICAL THREAD ERROR: {e}")
             self.is_running = False

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.thread = None
        print("System Audio Recording Stopped")

@st.cache_resource
def get_system_recorder():
    return SystemAudioRecorder()

system_recorder = get_system_recorder()


import wave

def process_audio_chunk(audio_data, sample_rate, client, model="whisper-1"):
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
            if client == "DEMO_MODE":
                return "Tell me about yourself (Demo Mode)"

            transcript = client.audio.transcriptions.create(
                model=model, 
                file=buffer,
                language="en" 
            )
            return transcript.text
    except Exception as e:
        error_msg = str(e)
        print(f"DEBUG: AI Generation Error: {error_msg}")
        return f"__API_ERROR__: {error_msg}"
        print(f"Processing error: {e}")
        
        if "insufficient_quota" in error_msg:
             return "__QUOTA_EXCEEDED__"
        
        # Return the actual error message for debugging
        return f"__API_ERROR__: {error_msg}"
    return ""

def generate_ai_response(transcript_text, context_text, client, model="gpt-4o", placeholder=None):
    if not transcript_text.strip():
        return None
        
    # Handle Demo Mode
    if client == "DEMO_MODE":
        time.sleep(1.0) # Simulate processing time
        
        # In Demo Mode, try to generate a somewhat relevant answer if context exists, 
        # otherwise use the generic placeholder.
        # Since we can't use GPT, we can only do simple keyword matching or return a better placeholder.
        
        return f"""[ANSWER]
(DEMO MODE - NO API KEY) 
**NOTE:** This is a canned placeholder because no API Key is active. 
To get REAL answers based on your uploaded Resume & Script, please enter your OpenAI API Key in the sidebar and disable Demo Mode.

(Example Response): I chose to leave my past job because I am looking for a new challenge where I can fully utilize my skills in [Your Main Skill]. While I learned a lot at [Previous Company], I am ready to take on more responsibility and deliver results for a team like yours.

[KEY POINTS]
- Seeking growth and new challenges
- Gratitude for past experience
- Eager to contribute to [Current Company]"""

    try:
        print(f"DEBUG: Generating AI Response using Model: {model}")
        
        # Simplified System Prompt for Faster Inference (especially for Llama3)
        system_prompt = """You are an expert interview candidate assistant.
Your goal is to generate a concise, HIGHLY PERSUASIVE, human-like response to the interviewer's question.
You are the candidate. Speak in the first person ("I", "me", "my").

CRITICAL GOAL: GET HIRED.
Every answer must demonstrate value, competence, and confidence. Make the interviewer feel like I am the perfect fit.

STRATEGY:
- **Be Result-Oriented**: State the IMPACT (e.g., "I led the migration, which cut costs by 15%").
- **Be Confident**: Use "I did," "I built," "I achieved."
- **Be Concise**: Short, punchy sentences.

CRITICAL CONTEXT INSTRUCTIONS:
1. **SCRIPT PRIORITY**: If the question matches the script, ADAPT THE SCRIPT'S ANSWER.
2. **RESUME ALIGNMENT**: If no script match, use facts from the "RESUME" section.
3. **JOB DESCRIPTION**: Use the "JD" to align keywords.

Rules:
1. **POV**: ALWAYS use "I".
2. **Tone**: Confident, conversational.
3. **Structure**: Direct Answer -> Example (STAR Method) -> Connection to Role.
4. **Length**: 30-60 seconds (approx 80-120 words).
5. **MANDATORY**: Include a metric/result.

Output Format:
[ANSWER]
[Your natural, conversational response here]

[KEY POINTS]
- Main achievement
- Relevant resume detail"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"My Resume, JD & Script Context:\n{context_text}\n\nInterviewer Question/Transcript:\n{transcript_text}"}
        ]
        
        # Force UI update to "Thinking..." before calling API
        if placeholder:
             update_hud(placeholder, f'<span class="transcript-label">Heard</span><br>{transcript_text}', '<span class="thinking">Generating answer... (Wait)</span>')

        # Use NON-STREAMING call for stability and speed debugging
        # Streaming often causes issues with certain providers or network latency in loops
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=300,
            temperature=0.7
        )
        
        full_response = response.choices[0].message.content
        print(f"DEBUG: AI Response Received: {len(full_response)} chars")
        return full_response
        
    except Exception as e:
        error_msg = str(e)
        print(f"DEBUG: AI Generation Error: {error_msg}")
        return f"__API_ERROR__: {error_msg}"

def update_hud(placeholder, status_html, answer_text, warning_html=""):
    if not placeholder: return
    
    # Minified HTML to avoid Markdown code block rendering issues
    hud_html = f"""<div id="live-answer-hud" class="floating-answer-box"><div class="hud-drag-handle"><span>Interview Copilot</span><span style="opacity: 0.5;">::</span></div><div class="hud-content"><div class="transcript-box">{warning_html}<div style="margin-top:5px;">{status_html}</div></div><div class="answer-box"><h4>Suggested Answer</h4><p>{answer_text}</p></div></div></div>"""
    placeholder.markdown(hud_html, unsafe_allow_html=True)

# --- UI ---

st.markdown("""
<style>
    .big-button button {
        width: 100%;
        height: 70px;
        font-size: 26px !important;
        font-weight: bold !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 14px 0 rgba(0,0,0,0.2) !important;
        transition: transform 0.1s ease-in-out;
    }
    .big-button button:hover {
        transform: scale(1.02);
    }
    .stAlert {
        border-radius: 12px;
    }
    /* Hide default streamlit menu for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* HUD Styles */
    .floating-answer-box {
        position: fixed;
        top: 20px;
        right: 20px;
        width: 450px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        z-index: 99999;
        border: 1px solid #e0e0e0;
        backdrop-filter: blur(10px);
        font-family: 'Inter', sans-serif;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .hud-drag-handle {
        padding: 8px 15px;
        background: #f8f9fa;
        cursor: move;
        border-bottom: 1px solid #eee;
        color: #888;
        font-size: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .hud-content {
        padding: 20px;
    }
    .transcript-box {
        margin-bottom: 15px;
        font-size: 14px;
        color: #555;
        line-height: 1.5;
        max-height: 100px;
        overflow-y: auto;
        padding-bottom: 10px;
        border-bottom: 1px solid #eee;
    }
    .transcript-label {
        display: block;
        font-size: 11px;
        color: #999;
        margin-bottom: 4px;
        text-transform: uppercase;
        font-weight: 700;
    }
    .answer-box h4 {
        margin: 0 0 10px 0;
        font-size: 16px;
        color: #2c3e50;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .answer-box p {
        margin: 0;
        font-size: 16px;
        color: #1a1a1a;
        line-height: 1.6;
        font-weight: 500;
    }
    .thinking {
        color: #666;
        font-style: italic;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
</style>

<script>
    // Simple Drag Logic for HUD
    const hud = window.parent.document.getElementById('live-answer-hud');
    if (hud) {
        const header = hud.querySelector('.hud-drag-handle');
        let isDragging = false;
        let startX, startY, initialLeft, initialTop;

        header.onmousedown = function(e) {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            initialLeft = hud.offsetLeft;
            initialTop = hud.offsetTop;
            hud.style.transition = 'none'; // Disable transition during drag
        };

        window.parent.document.onmousemove = function(e) {
            if (isDragging) {
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                hud.style.left = (initialLeft + dx) + 'px';
                hud.style.top = (initialTop + dy) + 'px';
                hud.style.right = 'auto'; // Clear right alignment
            }
        };

        window.parent.document.onmouseup = function() {
            isDragging = false;
            hud.style.transition = 'all 0.3s ease'; // Re-enable transition
        };
    }
</script>
""", unsafe_allow_html=True)

st.title("Interview Copilot")

# --- SIDEBAR ---
# --- SIDEBAR ---
with st.sidebar:
    st.header("Interview Copilot")
    
    # 0. Provider Selection
    st.markdown("### 🤖 AI Provider")
    provider_option = st.selectbox("Select Provider", ["OpenAI (Recommended)", "Groq (Free/Fast)"], index=0)
    
    # 1. API Key
    if "Groq" in provider_option:
        api_label = "Groq API Key"
        api_link = "https://console.groq.com/keys"
        env_key = os.getenv("GROQ_API_KEY", "")
    else:
        api_label = "OpenAI API Key"
        api_link = "https://platform.openai.com/api-keys"
        env_key = os.getenv("OPENAI_API_KEY", "")

    with st.expander(f"🔑 {api_label}", expanded=not env_key):
        api_key_input = st.text_input("Enter Key", type="password", value=env_key, label_visibility="collapsed")
        st.markdown(f"[Get {api_label}]({api_link})")
        
        if st.checkbox("Demo Mode (Free)", value=False):
            client = "DEMO_MODE"
            model_choice = "demo"
            transcription_model = "demo"
        elif api_key_input:
            if "Groq" in provider_option:
                os.environ["GROQ_API_KEY"] = api_key_input
                client = OpenAI(api_key=api_key_input, base_url="https://api.groq.com/openai/v1")
                model_choice = "llama3-70b-8192"
                transcription_model = "whisper-large-v3"
            else:
                os.environ["OPENAI_API_KEY"] = api_key_input
                client = OpenAI(api_key=api_key_input)
                model_choice = "gpt-4o"
                transcription_model = "whisper-1"
        else:
            client = None
            model_choice = None
            transcription_model = None

    # 2. Context Management (Grouped)
    st.markdown("### 📄 Context")
    st.caption("Upload materials for the AI to reference.")
    
    with st.container():
        resume_file = st.file_uploader("Resume (PDF/TXT)", type=["pdf", "docx", "txt"])
        job_desc = st.text_area("Job Description", height=100, placeholder="Paste JD here...")
        script_file = st.file_uploader("Script/Notes (Optional)", type=["pdf", "docx", "txt"])
        
        # Auto-save context on interaction is tricky in Streamlit, so we use a button but make it part of the flow
        if st.button("✅ Save & Load Context", use_container_width=True):
            r_text = get_text_from_upload(resume_file)
            s_text_file = get_text_from_upload(script_file)
            # We don't have script_text input anymore to simplify, or we can add it back if needed.
            # Let's keep it simple: File OR Text for script? 
            # User used text area before. Let's add it back but collapsed?
            # No, keep it simple. Just file for script is cleaner, but text is useful for quick notes.
            # Let's add a small text area for notes.
            
            full_script = s_text_file
            st.session_state['context_text'] = f"RESUME:\n{r_text}\n\nJD:\n{job_desc}\n\nSCRIPT:\n{full_script}"
            
            st.success("Context Loaded Successfully!")
            time.sleep(1)
            st.rerun()

    # 3. Advanced Settings (Hidden by default)
    st.markdown("---")
    with st.expander("⚙️ Advanced Settings"):
        st.subheader("Audio Source")
        # Loopback Devices
        loopback_devices = {}
        try:
            devices = system_recorder.get_loopback_devices()
            for d in devices:
                loopback_devices[d["name"]] = d["index"]
        except:
            pass
            
        if loopback_devices:
            default_name = list(loopback_devices.keys())[0]
            if "selected_device_name" in st.session_state and st.session_state.selected_device_name in loopback_devices:
                default_name = st.session_state.selected_device_name
            
            selected_device_name = st.selectbox("Input Device", list(loopback_devices.keys()), index=list(loopback_devices.keys()).index(default_name))
            
            if selected_device_name:
                target_idx = loopback_devices[selected_device_name]
                if system_recorder.device_index != target_idx:
                    system_recorder.set_device(target_idx)
                    st.session_state.selected_device_name = selected_device_name
        else:
            st.error("No Loopback Devices Found!")

        st.subheader("Preferences")
        silence_threshold_val = st.slider("Mic Sensitivity", 0.0001, 0.0100, 0.0020, 0.0001, format="%.4f")
        if st.button("Reset Transcript"):
            st.session_state["last_transcript"] = ""
            st.rerun()

# --- MAIN LAYOUT ---

# HUD (Always rendered but hidden via CSS if needed, or just rendered)
hud_id = "live-answer-hud"
if "last_transcript" in st.session_state and st.session_state["last_transcript"]:
    last_text = st.session_state["last_transcript"][-100:] + "..." if len(st.session_state["last_transcript"]) > 100 else st.session_state["last_transcript"]
else:
    last_text = "Listening..."
    
if "ai_answer" in st.session_state:
    current_answer = st.session_state.ai_answer
else:
    current_answer = "Ready..."

# Initial HUD State
st.markdown(f"""
<div id="{hud_id}" class="floating-answer-box">
    <div class="hud-drag-handle">
        <span>Interview Copilot</span>
        <span style="opacity: 0.5;">::</span>
    </div>
    <div class="hud-content">
        <div class="transcript-box">
            <span class="transcript-label">Heard</span>
            {last_text}
        </div>
        <div class="answer-box">
            <h4>Suggested Answer</h4>
            <p>{current_answer}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# DASHBOARD STATE
if system_recorder.is_running:
    # --- RUNNING ---
    st.markdown("### 🔴 Live Session Active")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        st.info(f"Listening on: **{selected_device_name if 'selected_device_name' in locals() else 'Default'}**")
        # Audio Meter
        rms_placeholder = st.progress(0.0)
        status_placeholder = st.empty()
        transcript_placeholder = st.empty()
        suggestion_placeholder = st.empty() # For HUD updates
        
    with c2:
        if st.button("⏹ STOP SESSION", type="primary", use_container_width=True):
            system_recorder.stop()
            st.rerun()

    is_playing = True

else:
    # --- STOPPED ---
    is_playing = False
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## Ready to Interview?")
    
    # Status Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. API Key**")
        if client:
            st.success("Connected")
        else:
            st.error("Missing")
            
    with col2:
        st.markdown("**2. Context**")
        ctx_len = len(st.session_state.get('context_text', ''))
        if ctx_len > 50:
            st.success(f"Loaded ({ctx_len} chars)")
        else:
            st.warning("Not Loaded")
            
    with col3:
        st.markdown("**3. Audio**")
        if system_recorder.device_index is not None:
             st.success("Device Selected")
        else:
             st.warning("Auto-Scan Ready")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # HERO BUTTON
    st.markdown('<div class="big-button">', unsafe_allow_html=True)
    if st.button("▶ START INTERVIEW SESSION", type="primary"):
        # Validation
        if not client:
            st.toast("Please enter API Key first!", icon="🔑")
        else:
            if system_recorder.device_index is None:
                system_recorder.auto_scan_and_start(start_immediately=True)
            else:
                system_recorder.start()
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("🎧 Auto-connects to System Audio (Zoom, Meet, Teams). No login required.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Troubleshooting"):
         col_t1, col_t2 = st.columns(2)
         with col_t1:
             if st.button("🔊 Test Audio (3s)"):
                 found_dev = system_recorder.auto_scan_and_start(start_immediately=False)
                 if found_dev:
                     st.success(f"✅ Scan Complete. Selected: **{found_dev}**")
         
         with col_t2:
             if st.button("🛠️ Run Full Diagnostic"):
                 if not client:
                     st.error("Please enter API Key first.")
                 else:
                     with st.spinner("Running System Diagnostic..."):
                         # 1. Test Audio Capture
                         st.write("1. Testing Audio Capture (3s)... Play Sound NOW!")
                         
                         test_pa = pyaudio.PyAudio()
                         try:
                             # Get device info for rate
                             dev_idx = system_recorder.device_index if system_recorder.device_index is not None else None
                             if dev_idx is None:
                                  # Try default
                                  dev_info = test_pa.get_default_input_device_info()
                                  dev_idx = dev_info['index']
                             
                             dev_info = test_pa.get_device_info_by_index(dev_idx)
                             native_rate = int(dev_info.get('defaultSampleRate', 44100))
                             st.write(f"   > Device: `{dev_info['name']}` | Rate: `{native_rate}`")

                             stream = test_pa.open(
                                 format=pyaudio.paFloat32,
                                 channels=1,
                                 rate=native_rate, # USE NATIVE RATE
                                 input=True,
                                 input_device_index=dev_idx,
                                 frames_per_buffer=1024
                             )
                             
                             frames = []
                             rms_vals = []
                             # Read 3 seconds
                             for _ in range(0, int(native_rate / 1024 * 3)):
                                 data = stream.read(1024, exception_on_overflow=False)
                                 frames.append(data)
                                 audio_data = np.frombuffer(data, dtype=np.float32)
                                 rms_vals.append(np.sqrt(np.mean(audio_data**2)))
                             
                             stream.stop_stream()
                             stream.close()
                             
                             avg_rms = sum(rms_vals) / len(rms_vals)
                             max_rms = max(rms_vals)
                             st.write(f"   > Avg RMS: `{avg_rms:.6f}` | Max RMS: `{max_rms:.6f}`")
                             
                             if max_rms < 0.001:
                                 st.error("❌ Audio is SILENT. Check Device/Volume.")
                             else:
                                 st.success("✅ Audio Capture OK")
                                 
                                 # 2. Test Transcription
                                 st.write("2. Testing OpenAI Whisper API...")
                                 full_audio = b''.join(frames)
                                 audio_np = np.frombuffer(full_audio, dtype=np.float32)
                                 
                                 # Resample to 16000 for Whisper if needed
                                 if native_rate != 16000:
                                     num_samples = int(len(audio_np) * 16000 / native_rate)
                                     audio_np = scipy.signal.resample(audio_np, num_samples)

                                 # Convert to Int16 WAV for API
                                 import io
                                 import wave
                                 wav_io = io.BytesIO()
                                 with wave.open(wav_io, 'wb') as wf:
                                     wf.setnchannels(1)
                                     wf.setsampwidth(2) # 16-bit
                                     wf.setframerate(16000)
                                     # Convert float32 -> int16
                                     int16_data = (audio_np * 32767).astype(np.int16)
                                     wf.writeframes(int16_data.tobytes())
                                 
                                 wav_io.seek(0)
                                 wav_io.name = "diagnostic.wav"
                                 
                                 try:
                                     transcript = client.audio.transcriptions.create(
                                         model="whisper-1",
                                         file=wav_io,
                                         language="en"
                                     )
                                     st.write(f"   > Transcript: '{transcript.text}'")
                                     st.success("✅ API Connection OK")
                                 except Exception as e:
                                     error_msg = str(e)
                                     if "insufficient_quota" in error_msg:
                                         st.error("❌ API Failed: QUOTA EXCEEDED. Check billing at platform.openai.com")
                                     else:
                                         st.error(f"❌ API Failed: {e}")
                                     
                         except Exception as e:
                             st.error(f"❌ Diagnostic Error: {e}")
                         finally:
                             test_pa.terminate()

# --- AUDIO PROCESSING LOOP (Background) ---
if is_playing:
    # Buffering Logic
    speech_buffer = []
    silence_frames = 0
    MIN_CHUNKS_TO_PROCESS = 2 # Reduced from 3 to capture short "Yes/No"
    MAX_BUFFER_SIZE = 60 # Reduced from 150 to 60 (6s) to prevent "Listening" forever loop
    SILENCE_THRESHOLD = silence_threshold_val 
    SILENCE_DURATION_TRIGGER = 4 # Reduced from 5 to 4 (0.4s) for snappier response 
    
    # Check queue
    while is_playing:
        try:
            # Non-blocking get
            audio_data, rate = audio_queue.get_nowait()
            
            # RMS for UI
            rms = 0.0
            if isinstance(audio_data, np.ndarray):
                # Ensure float32 for calculation to prevent overflow
                audio_float = audio_data.astype(np.float32)
                rms = np.sqrt(np.mean(audio_float**2))
                
                # Boost it for visibility (Auto-Gain Visualization)
                display_level = min(rms * 500, 1.0) 
                rms_placeholder.progress(float(display_level))
                
                # --- DYNAMIC STATUS UPDATE ---
                status_color = "🟢" if rms > SILENCE_THRESHOLD else "⚪"
                status_msg = "LISTENING" if rms > SILENCE_THRESHOLD else "WAITING"
                if 'status_placeholder' in locals():
                     status_placeholder.markdown(f"**RMS:** `{rms:.6f}` | **Threshold:** `{SILENCE_THRESHOLD}` | {status_color} {status_msg}")
                
                # --- UPDATE HUD REAL-TIME ---
                # We update the 'processing_status' part of the HUD
                if rms > SILENCE_THRESHOLD:
                    new_status_text = f"Listening... (RMS: {rms:.4f})"
                else:
                    new_status_text = "Waiting..."
                    
                if 'suggestion_placeholder' in locals():
                    update_hud(suggestion_placeholder, f'<span class="transcript-label">Status</span><br>{new_status_text}', current_answer)
                
            # --- BUFFERING & SILENCE DETECTION ---
            speech_buffer.append(audio_data)
            
            # CHECK MAX BUFFER SIZE (Force Send)
            if len(speech_buffer) > MAX_BUFFER_SIZE:
                 rms = 0.0
                 silence_frames = SILENCE_DURATION_TRIGGER + 10
            
            # CHECK FORCE FLAG
            if st.session_state.get("force_transcribe", False):
                 st.session_state["force_transcribe"] = False # Reset
                 rms = 0.0 
                 silence_frames = SILENCE_DURATION_TRIGGER + 1
                 if len(speech_buffer) < MIN_CHUNKS_TO_PROCESS:
                     speech_buffer.extend([np.zeros_like(audio_data)] * 5)
            
            # --- HUD AUDIO METER ---
            if int(time.time() * 10) % 2 == 0: # Update every ~200ms
                mic_status_icon = "🟢" if rms > SILENCE_THRESHOLD else "⚪"
                # Visual Boost for HUD Bar
                mic_level_bar = "I" * int(rms * 1000) 
                if len(mic_level_bar) > 15: mic_level_bar = mic_level_bar[:15]
                
                # Check for "DEAD AUDIO" (No capture for 5+ seconds)
                if rms == 0.0:
                    st.session_state["zero_audio_frames"] = st.session_state.get("zero_audio_frames", 0) + 1
                else:
                    st.session_state["zero_audio_frames"] = 0
                
                # If 5 seconds of PURE ZERO (not just silence, but 0.0 data), warn user
                warning_html = ""
                if st.session_state.get("zero_audio_frames", 0) > 50: # 50 * 0.1s = 5s
                    warning_html = """
<div style="background: #ff4444; color: white; padding: 8px; border-radius: 4px; font-size: 0.8em; margin-bottom: 10px;">
    ⚠️ <b>NO AUDIO DETECTED!</b><br/>
    Check 'Audio Source' in Sidebar.
</div>
"""
                
                if 'suggestion_placeholder' in locals():
                    status_content = f'<span class="transcript-label">Status</span><br>{mic_status_icon} Listening <span style="color:#4CAF50; font-weight:bold;">{mic_level_bar}</span>'
                    update_hud(suggestion_placeholder, status_content, current_answer, warning_html)

            if rms < SILENCE_THRESHOLD:
                silence_frames += 1
            else:
                silence_frames = 0
            
            # Decide whether to process
            should_process = False
            
            # Trigger 1: Enough audio AND Silence detected (End of sentence)
            if len(speech_buffer) > MIN_CHUNKS_TO_PROCESS and silence_frames > SILENCE_DURATION_TRIGGER:
                should_process = True
                
            # Trigger 2: Buffer too full (Force flush to avoid memory issues/too long latency)
            if len(speech_buffer) > MAX_BUFFER_SIZE:
                should_process = True
            
            if not should_process:
                # If we are just accumulating silence at the start, keep buffer clean
                if len(speech_buffer) > 0 and len(speech_buffer) < MIN_CHUNKS_TO_PROCESS and silence_frames == len(speech_buffer):
                     speech_buffer = []
                     silence_frames = 0
                continue

            # --- PROCESS BUFFER ---
            full_audio = np.concatenate(speech_buffer)
            
            # Check average RMS
            phrase_rms = np.sqrt(np.mean(full_audio**2))
            
            # Reset Buffer
            speech_buffer = []
            silence_frames = 0
            
            if phrase_rms < 0.001:
                continue
            
            if not client:
                st.toast("⚠️ Audio detected, but NO API KEY.", icon="🛑")
                continue

            # Show "Voice Detected" temporarily
            if 'suggestion_placeholder' in locals():
                update_hud(suggestion_placeholder, '<span class="transcript-label">Status</span><br>🎤 Voice Detected... Processing...', current_answer)

            # Process
            text = process_audio_chunk(full_audio, rate, client, transcription_model)
            
            if text == "__QUOTA_EXCEEDED__":
                 st.error("🚨 CRITICAL: OpenAI API Quota Exceeded. Please check your billing at platform.openai.com.")
                 st.toast("🚨 Quota Exceeded! Stopping Session.", icon="🛑")
                 is_playing = False
                 st.session_state['run_pipeline'] = False
                 break
            
            if text and text.startswith("__API_ERROR__"):
                error_detail = text.replace("__API_ERROR__: ", "")
                st.toast(f"⚠️ API Error: {error_detail[:50]}...", icon="❌")
                print(f"Full API Error: {error_detail}")
                
                if 'suggestion_placeholder' in locals():
                    update_hud(suggestion_placeholder, f'<span style="color:red">API ERROR</span>', f"Details: {error_detail}")
                continue

            if text is None:
                st.toast("⚠️ Transcription Failed.", icon="❌")
                continue

            # --- TRANSCRIPT FILTERING ---
            IGNORED_PHRASES = ["Thank you.", "Bye.", "Silence.", "MBC News", "You", "Unclear"]
            
            if text and len(text.strip()) > 5 and text.strip() not in IGNORED_PHRASES:
                print(f"Transcribed: {text}")
                current_transcript = st.session_state["last_transcript"] + " " + text
                st.session_state["last_transcript"] = current_transcript[-2000:] 
                
                if 'transcript_placeholder' in locals():
                    transcript_placeholder.markdown(f"**Transcript:**\n\n{st.session_state['last_transcript']}")
                
                # Show "Thinking..."
                if 'suggestion_placeholder' in locals():
                    update_hud(suggestion_placeholder, f'<span class="transcript-label">Heard</span><br>{text}', '<span class="thinking">Generating answer...</span>')
                
                # Generate AI Response
                context = st.session_state.get('context_text', "No context loaded.")
                ai_answer = generate_ai_response(text, context, client, model_choice, placeholder=suggestion_placeholder)
                
                if ai_answer:
                    if "API KEY ERROR" in ai_answer:
                        st.error(ai_answer)
                    
                    st.session_state.ai_answer = ai_answer
                    if 'suggestion_placeholder' in locals():
                        update_hud(suggestion_placeholder, f'<span class="transcript-label">Heard</span><br>{text}', ai_answer)
                    
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in main loop: {e}")
            break
