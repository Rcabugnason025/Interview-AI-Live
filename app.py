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
        self.stream = None
        self.is_running = False
        self.pa = pyaudio.PyAudio()
        self.native_rate = 16000 # Default fallback
        self.native_channels = 1
        self.device_index = None # Manually selected device index

    def set_device(self, index):
        self.device_index = index

    def start(self):
        if self.is_running:
            return
            
        try:
            device_info = None
            
            # Use manually selected device if available
            if self.device_index is not None:
                device_info = self.pa.get_device_info_by_index(self.device_index)
            else:
                # Auto-detect logic (fallback)
                wasapi_info = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_speakers = self.pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                
                if not default_speakers["isLoopbackDevice"]:
                    found = False
                    for loopback in self.pa.get_loopback_device_info_generator():
                        # IMPROVED MATCHING: Check if name is contained
                        if default_speakers["name"] in loopback["name"]:
                            device_info = loopback
                            found = True
                            break
                    if not found:
                        try:
                            device_info = next(self.pa.get_loopback_device_info_generator())
                        except StopIteration:
                            raise Exception("No loopback device found.")
                else:
                    device_info = default_speakers
            
            print(f"Recording from: {device_info['name']}")
            
            # WASAPI Loopback requires matching the device's native sample rate and channel count
            self.native_rate = int(device_info["defaultSampleRate"])
            self.native_channels = int(device_info["maxInputChannels"]) 
            
            print(f"Device Native Rate: {self.native_rate}, Channels: {self.native_channels}")

            # Open stream with paFloat32 (usually required for WASAPI loopback)
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self.native_channels,
                rate=self.native_rate,
                input=True,
                input_device_index=device_info["index"],
                frames_per_buffer=int(self.native_rate * 0.1), # 100ms buffer
                stream_callback=self._callback
            )
            
            self.stream.start_stream()
            self.is_running = True
            print("System Audio Recording Started via PyAudioWpatch")

        except Exception as e:
            st.error(f"""
            **System Audio Capture Failed** ({e})
            
            **👇 HOW TO FIX:**
            1. Go to the **Sidebar** -> **Audio Settings**.
            2. Under **"Select Speaker Device"**, choose a different device.
            3. Try **Headphones** or **Speakers** explicitly.
            4. Click "Stop Listening" then "Start Listening" again.
            
            *Falling back to default microphone...*
            """)
            try:
                self.fallback_stream = sd.InputStream(
                    callback=audio_callback,
                    channels=1,
                    samplerate=16000
                )
                self.fallback_stream.start()
                self.is_running = True
            except Exception as e2:
                st.error(f"Failed to start microphone: {e2}")

    def _callback(self, in_data, frame_count, time_info, status):
        # Convert raw bytes to numpy array (Float32)
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Reshape to (frames, channels)
        if self.native_channels > 1:
            audio_data = audio_data.reshape(-1, self.native_channels)
            # Mix to mono (average)
            audio_mono = audio_data.mean(axis=1)
        else:
            audio_mono = audio_data
            
        # Resample if necessary (native_rate -> 16000)
        target_rate = 16000
        if self.native_rate != target_rate:
            # Calculate number of samples
            num_samples = int(len(audio_mono) * target_rate / self.native_rate)
            audio_resampled = scipy.signal.resample(audio_mono, num_samples)
        else:
            audio_resampled = audio_mono
            
        # Normalize/Clamp (already float32, but just in case)
        # audio_float = audio_resampled # It's already float -1.0 to 1.0
        
        # Reshape to (N, 1)
        audio_float = audio_resampled.reshape(-1, 1).astype(np.float32)
        
        audio_queue.put(audio_float)
        return (in_data, pyaudio.paContinue)


    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if hasattr(self, 'fallback_stream') and self.fallback_stream:
            self.fallback_stream.stop()
            self.fallback_stream.close()
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

def generate_ai_response(transcript_text, context_text, client, model="gpt-4o"):
    if not transcript_text.strip():
        return None
        
    # Handle Demo Mode
    if client == "DEMO_MODE":
        time.sleep(1.5) # Simulate processing time
        return f"""[ANSWER]
(DEMO MODE) That's a great question. Based on my experience, I believe my background in [Skill from Resume] makes me a strong fit. I have successfully handled similar situations by prioritizing clear communication and strategic planning.

[KEY POINTS]
- Demonstrated [Skill]
- Referenced previous role at [Company]
- Showcased problem-solving ability"""

    try:
        messages = [
            {"role": "system", "content": f"""You are an expert interview candidate assistant.
Your goal is to generate a concise, HIGHLY PERSUASIVE, human-like response to the interviewer's question.
You are the candidate. Speak in the first person ("I", "me", "my").

CRITICAL GOAL: GET HIRED.
Every answer must demonstrate value, competence, and confidence. Make the interviewer feel like I am the perfect fit.

CRITICAL CONTEXT INSTRUCTIONS:
1. **SCRIPT PRIORITY**: If the question matches (even partially) anything in the "SCRIPT" section below, YOU MUST USE THE SCRIPT'S ANSWER. Adapt it slightly to sound natural, but keep the core content.
2. **RESUME ALIGNMENT**: If no script match, build the answer using ONLY facts from the "RESUME" section.
3. **JOB DESCRIPTION**: Use the "JD" section to align your keywords, but do not invent experience.

Rules:
1. **POV**: ALWAYS use "I". Never say "The candidate" or "Based on the resume".
2. **Tone**: Confident, Action-Oriented, Professional. Use strong verbs (e.g., "I spearheaded," "I delivered," "I optimized").
3. **Structure**: Direct Answer -> Example (STAR Method) -> Connection to Role.
4. **Length**: 30-60 seconds speaking time (approx 80-120 words).
5. **MANDATORY**: Include a specific metric or result from the Resume/Script if possible (e.g., "increased sales by 20%").
6. If I lack direct experience, pivot to transferable skills and express eagerness to apply them to this role.

Output Format:
[ANSWER]
[Your natural, conversational response here in first person]

[KEY POINTS]
- Main achievement/skill mentioned
- Relevant resume detail referenced"""},
            {"role": "user", "content": f"My Resume, JD & Script Context:\n{context_text}\n\nInterviewer Question/Transcript:\n{transcript_text}"}
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
             return "⚠️ API KEY ERROR: Please check the API Key in the sidebar. It seems incorrect or expired."
        return f"AI Error: {e}"

# --- UI ---

st.title("🎤 Live Interview AI Copilot")

# Sidebar
with st.sidebar:
    st.subheader("Configuration")
    
    # Model Selection
    model_choice = st.selectbox("AI Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], index=0)
    
    # API Key Input with cleanup
    api_key_input = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    api_key = api_key_input.strip() if api_key_input else None
    
    # Connection Test
    if api_key and st.button("🔌 Test Connection"):
        try:
            test_client = OpenAI(api_key=api_key)
            test_client.models.list()
            st.success("API Key is Valid! ✅")
        except Exception as e:
            st.error(f"API Key Invalid: {e}")

    # Toggle for Demo Mode
    demo_mode = st.checkbox("Enable Demo Mode (No API Key needed)", value=False)
    
    if demo_mode:
        client = "DEMO_MODE"
        st.info("Demo Mode Enabled: AI will respond with placeholder text.")
    elif api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        client = OpenAI(api_key=api_key)
    else:
        st.warning("Enter OpenAI API Key or Enable Demo Mode")
        client = None
    
    st.subheader("Audio Settings")
    audio_source = st.radio("Audio Source", ["Browser Microphone (WebRTC)", "System Audio (Windows Loopback)"], index=1)
    st.caption("Use 'System Audio' to capture the Interviewer's voice from your speakers.")
    
    if audio_source == "System Audio (Windows Loopback)":
        # Get Loopback Devices
        loopback_devices = {}
        try:
            for loopback in system_recorder.pa.get_loopback_device_info_generator():
                loopback_devices[loopback["name"]] = loopback["index"]
        except Exception as e:
            st.error(f"Error listing devices: {e}")
            
        if loopback_devices:
            # Try to pre-select default
            default_name = list(loopback_devices.keys())[0]
            # Try to smart match system default
            try:
                wasapi_info = system_recorder.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
                sys_default = system_recorder.pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                for name in loopback_devices:
                    if sys_default["name"] in name:
                        default_name = name
                        break
            except:
                pass
            
            selected_device_name = st.selectbox("Select Speaker Device", list(loopback_devices.keys()), index=list(loopback_devices.keys()).index(default_name) if default_name in loopback_devices else 0)
            system_recorder.set_device(loopback_devices[selected_device_name])
        else:
            st.warning("No System Audio devices found.")

    st.markdown("### Audio Levels")
    rms_placeholder = st.empty()
    status_placeholder = st.empty()

    st.markdown("---")
    st.subheader("🧪 Manual Test Mode")
    st.caption("Type a question below to test AI answers without audio.")
    test_question = st.text_input("Simulate Interviewer Question", placeholder="e.g. Tell me about yourself")
    if st.button("Generate Test Answer"):
        if test_question:
            # Update transcript for history
            st.session_state["last_transcript"] = st.session_state.get("last_transcript", "") + " [TEST]: " + test_question
            
            # Generate Response
            context = st.session_state.get('context_text', "No context loaded.")
            # Show a temporary status (will be overwritten on rerun)
            st.toast("Generating answer...")
            
            ai_answer = generate_ai_response(test_question, context, client, model_choice)
            
            if ai_answer:
                if "API KEY ERROR" in ai_answer:
                     st.error(ai_answer)
                st.session_state.ai_answer = ai_answer
                st.rerun()
        else:
            st.warning("Please type a question first.")

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

    # --- HUD Settings ---
    st.markdown("---")
    with st.sidebar.expander("🎨 HUD / Overlay Settings", expanded=True):
        st.caption("Customize the Live Answer Box position and size.")
        
        col_pos1, col_pos2 = st.columns(2)
        with col_pos1:
            hud_y_pos = st.slider("Vertical (Top %)", 0, 100, 10, help="0% = Top, 100% = Bottom")
        with col_pos2:
            hud_x_pos = st.slider("Horizontal (Left %)", 0, 100, 50, help="0% = Left, 100% = Right")
            
        col_size1, col_size2 = st.columns(2)
        with col_size1:
            hud_width = st.slider("Width (px)", 300, 1000, 500)
        with col_size2:
            hud_opacity = st.slider("Opacity", 0.1, 1.0, 0.95)
            
        hud_font_size = st.slider("Font Size (px)", 12, 24, 16)
        
        # Calculate CSS positioning
        # We use 'top' and 'left' percentages. 
        # To center perfectly when 50%, we use transform.
        css_pos = f"""
            top: {hud_y_pos}%;
            left: {hud_x_pos}%;
            transform: translate(-{hud_x_pos}%, 0);
        """

    if st.button("Clear Transcript"):
        st.session_state["last_transcript"] = ""
        st.rerun()
        
    # --- Live Interview HUD (Custom CSS) ---
    # Enhanced CSS for "Movable" look and "Always on Top"
    st.markdown(f"""
    <style>
        .floating-answer-box {{
            position: fixed;
            {css_pos}
            width: {hud_width}px;
            max-height: 80vh;
            overflow-y: auto;
            background-color: rgba(20, 20, 20, {hud_opacity});
            color: #e0e0e0;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            z-index: 999999; /* Super high z-index to overwrite everything */
            font-family: 'Segoe UI', sans-serif;
            font-size: {hud_font_size}px;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }}
        .transcript-box {{
            font-size: 0.85em;
            color: #aaa;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #444;
            font-style: italic;
        }}
        .answer-box h4 {{
            margin: 0 0 8px 0;
            font-size: 1em;
            color: #4CAF50;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .answer-box p {{
            margin: 0;
            line-height: 1.5;
            white-space: pre-wrap; /* Preserve newlines */
        }}
        /* Custom Scrollbar */
        .floating-answer-box::-webkit-scrollbar {{
            width: 8px;
        }}
        .floating-answer-box::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.05);
        }}
        .floating-answer-box::-webkit-scrollbar-thumb {{
            background: #555;
            border-radius: 4px;
        }}
        .floating-answer-box::-webkit-scrollbar-thumb:hover {{
            background: #777;
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
        if st.button("Start Listening", type="primary"):
            try:
                system_recorder.start()
                st.rerun()
            except Exception as e:
                st.error(f"""
                **Could not start System Audio Capture.**
                
                Error details: `{e}`
                
                **👇 HOW TO FIX:**
                1. Look at the **Sidebar** on the left.
                2. Find **"Audio Source"**.
                3. Under **"Select Speaker Device"**, try choosing a different device (e.g., "Headphones" or "Speakers").
                4. Click "Start Listening" again.
                """)
        
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

    # Always render the HUD if there is an existing answer/transcript in session state
    # This ensures the HUD is visible even if we are not currently processing a chunk
    # Or if we just started listening
    if "last_transcript" in st.session_state and st.session_state["last_transcript"]:
        last_text = st.session_state["last_transcript"][-100:] + "..." if len(st.session_state["last_transcript"]) > 100 else st.session_state["last_transcript"]
    else:
        last_text = "Listening..."
        
    if "ai_answer" in st.session_state:
        current_answer = st.session_state.ai_answer
    else:
        current_answer = "Waiting for question..."
    
    # --- PROCESSING INDICATOR ---
    # Add a visual cue if audio is detected but not yet transcribed
    processing_status = ""
    
    # Initial Render of HUD
    suggestion_placeholder.markdown(f"""
    <div class="floating-answer-box">
        <div class="transcript-box">
            <span class="label">Status:</span> {last_text} {processing_status}
        </div>
        <div class="answer-box">
            <h4>Live Answer:</h4>
            <p>{current_answer}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main Loop for processing
if is_playing and client:
    # We check the queue periodically
    while is_playing:
        try:
            # Non-blocking get
            audio_data, rate = audio_queue.get_nowait()
            
            # Calculate RMS for visual feedback
            rms = 0.0
            if isinstance(audio_data, np.ndarray):
                # Ensure float32 for calculation to prevent overflow
                audio_float = audio_data.astype(np.float32)
                rms = np.sqrt(np.mean(audio_float**2))
                
                # Boost it for visibility
                level = min(rms * 10, 1.0) 
                rms_placeholder.progress(float(level))
                status_placeholder.text(f"Level: {level:.3f}")
            
            # --- SILENCE DETECTION ---
            # Threshold: 0.01 is a reasonable starting point. 
            # If RMS is too low, skip processing to save API calls and reduce latency.
            SILENCE_THRESHOLD = 0.005 
            
            if rms < SILENCE_THRESHOLD:
                # Skip processing this chunk
                continue
            
            # Show "Voice Detected" temporarily
            suggestion_placeholder.markdown(f"""
            <div class="floating-answer-box">
                <div class="transcript-box">
                    <span class="label">Status:</span> 🎤 Voice Detected... Processing...
                </div>
                <div class="answer-box">
                    <h4>Live Answer:</h4>
                    <p>{current_answer}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Process
            text = process_audio_chunk(audio_data, rate, client)
            
            # --- TRANSCRIPT FILTERING ---
            # Filter out common Whisper hallucinations or very short noise
            IGNORED_PHRASES = ["Thank you.", "Bye.", "Silence.", "MBC News", "You", "Unclear"]
            
            if text and len(text.strip()) > 5 and text.strip() not in IGNORED_PHRASES:
                print(f"Transcribed: {text}")
                # Update transcript
                current_transcript = st.session_state["last_transcript"] + " " + text
                st.session_state["last_transcript"] = current_transcript[-2000:] # Keep last 2000 chars
                
                # Show transcript in main view (optional, but good for history)
                transcript_placeholder.markdown(f"**Transcript:**\n\n{st.session_state['last_transcript']}")
                
                # UPDATE HUD IMMEDIATELY with the new transcript (so user knows it's being processed)
                # We show the last AI answer while calculating the new one, or a "Thinking..." status
                suggestion_placeholder.markdown(f"""
                <div class="floating-answer-box">
                    <div class="transcript-box">
                        <span class="label">Heard:</span> {text}
                    </div>
                    <div class="answer-box">
                        <span class="thinking">Generating answer...</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate AI Response
                context = st.session_state.get('context_text', "No context loaded.")
                ai_answer = generate_ai_response(text, context, client, model_choice)
                
                if ai_answer:
                    if "API KEY ERROR" in ai_answer:
                        st.error(ai_answer)
                    
                    st.session_state.ai_answer = ai_answer
                    # Update HUD with the Answer
                    suggestion_placeholder.markdown(f"""
                    <div class="floating-answer-box">
                        <div class="transcript-box">
                            <span class="label">Heard:</span> {text}
                        </div>
                        <div class="answer-box">
                            <h4>AI Suggested Answer:</h4>
                            <p>{ai_answer}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in main loop: {e}")
            break
