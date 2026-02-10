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

    def get_loopback_devices(self):
        """Returns a list of available loopback devices."""
        devices = []
        try:
            if hasattr(self.pa, 'get_loopback_device_info_generator'):
                for loopback in self.pa.get_loopback_device_info_generator():
                    devices.append({
                        "index": loopback["index"],
                        "name": loopback["name"]
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
        max_rms = 0.0
        
        # Temporary PyAudio instance for testing
        temp_pa = pyaudio.PyAudio()
        
        for dev in devices:
            try:
                print(f"Testing: {dev['name']} (Index {dev['index']})")
                
                # Try to open stream
                stream = temp_pa.open(
                    format=pyaudio.paFloat32,
                    channels=2, # Assume stereo
                    rate=44100, # Standard
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
                
                if avg_rms > max_rms:
                    max_rms = avg_rms
                    best_device = dev
                
                stream.stop_stream()
                stream.close()
                
            except Exception as e:
                print(f"  > Failed: {e}")
        
        temp_pa.terminate()
        
        if best_device: # Even if RMS is low, pick the one that didn't crash? 
             # Or only if RMS > 0.0001?
             # Let's pick the best one we found.
             print(f"Selected Best Device: {best_device['name']} (RMS: {max_rms:.6f})")
             self.set_device(best_device['index'])
             
             if start_immediately:
                 self.start()
                 
             return best_device['name']
        
        st.error("No working loopback devices found. Ensure audio is playing!")
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
            if client == "DEMO_MODE":
                return "Tell me about yourself (Demo Mode)"

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
        messages = [
            {"role": "system", "content": f"""You are an expert interview candidate assistant.
Your goal is to generate a concise, HIGHLY PERSUASIVE, human-like response to the interviewer's question.
You are the candidate. Speak in the first person ("I", "me", "my").

CRITICAL GOAL: GET HIRED.
Every answer must demonstrate value, competence, and confidence. Make the interviewer feel like I am the perfect fit.

STRATEGY:
- **Be Result-Oriented**: Don't just list tasks. State the IMPACT (e.g., "I led the migration, which cut costs by 15%").
- **Be Confident**: Avoid weak language like "I believe," "I think," or "I tried." Use "I did," "I built," "I achieved."
- **Be Concise**: Get to the point. Short, punchy sentences are better than long, winding ones.

CRITICAL CONTEXT INSTRUCTIONS (Strict Priority):
1. **SCRIPT PRIORITY (Highest)**: Check the "SCRIPT" section first. If the question matches (even partially) anything in the script, YOU MUST ADAPT THE SCRIPT'S ANSWER. Do not copy it robotically—rephrase it slightly to sound natural and conversational, but keep the core points and metrics.
2. **RESUME ALIGNMENT (Second)**: If no script match, build the answer using ONLY facts from the "RESUME" section. Highlight specific achievements.
3. **JOB DESCRIPTION (Third)**: Use the "JD" section to align your keywords and focus, but do not invent experience not in the Resume/Script.

Rules for "Humanizing" & POV:
1. **POV**: ALWAYS use "I". Never say "The candidate" or "Based on the resume".
2. **Tone**: Confident, conversational, and authentic. Avoid overly formal or robotic AI language (e.g., avoid "I would be delighted to..."). instead say "I'd love to..." or "I'm really excited about...".
3. **Structure**: Direct Answer -> Example (STAR Method) -> Connection to Role.
4. **Length**: 30-60 seconds speaking time (approx 80-120 words).
5. **MANDATORY**: Include a specific metric or result from the Resume/Script if possible (e.g., "increased sales by 20%").
6. **Fillers**: It is okay to start with natural openers like "That's a great question," or "I'm glad you asked that," to buy time and sound human.

Output Format:
[ANSWER]
[Your natural, conversational response here in first person]

[KEY POINTS]
- Main achievement/skill mentioned
- Relevant resume detail referenced"""},
            {"role": "user", "content": f"My Resume, JD & Script Context:\n{context_text}\n\nInterviewer Question/Transcript:\n{transcript_text}"}
        ]
        
        # STREAMING RESPONSE
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                
                # Update UI Real-time if placeholder provided
                if placeholder:
                    placeholder.markdown(f"""
<div class="floating-answer-box">
<div class="transcript-box">
<span class="label">Heard:</span> {transcript_text}
</div>
<div class="answer-box">
<h4>AI Suggested Answer:</h4>
<p>{full_response} ▌</p>
</div>
</div>
""", unsafe_allow_html=True)
        
        return full_response
        
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
    api_key_input = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""), help="Starts with 'sk-...'")
    st.markdown("[👉 Get an OpenAI API Key here](https://platform.openai.com/api-keys)")
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
        st.info("Demo Mode Enabled: AI will respond with placeholder text (No GPT).")
    elif api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        client = OpenAI(api_key=api_key)
    else:
        st.warning("⚠️ Enter OpenAI API Key to get REAL answers based on your Resume.")
        client = None
    
    st.subheader("Audio Settings")
    audio_source = st.radio("Audio Source", ["Browser Microphone (WebRTC)", "System Audio (Windows Loopback)"], index=1)
    st.caption("Use 'System Audio' to capture the Interviewer's voice from your speakers.")

    # --- SENSITIVITY SLIDER ---
    st.markdown("**Microphone Sensitivity**")
    silence_threshold_val = st.slider("Silence Threshold (RMS)", min_value=0.0001, max_value=0.0100, value=0.0020, step=0.0001, format="%.4f", help="Adjust this ABOVE your background noise level. If too low, it never stops listening.")
    
    # Troubleshooting Guide
    with st.expander("🛠️ Audio Troubleshooting", expanded=False):
        st.markdown("""
        **If Audio Level stays at 0.000:**
        1. **Ensure Audio is Playing:** Play a YouTube video or have someone speak on the call.
        2. **Try 'Auto-Scan Audio':** Click the button in the main screen to automatically find the active device.
        3. **Disable Exclusive Mode:**
           - Right-click Sound Icon in taskbar -> Sound Settings -> More Sound Settings.
           - Right-click your device -> Properties -> Advanced.
           - **Uncheck** "Allow applications to take exclusive control".
        4. **Check Device Name:** If using a headset, try selecting "Headphones" instead of "Headset Earphone".
        """)

    if audio_source == "System Audio (Windows Loopback)":
        # Get Loopback Devices
        loopback_devices = {}
        try:
            devices = system_recorder.get_loopback_devices()
            for d in devices:
                loopback_devices[d["name"]] = d["index"]
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
                    if sys_default["name"] in name or name in sys_default["name"]:
                        default_name = name
                        break
            except:
                pass
            
            selected_device_name = st.selectbox("Select Speaker Device", list(loopback_devices.keys()), index=list(loopback_devices.keys()).index(default_name) if default_name in loopback_devices else 0)
            
            # FORCE UPDATE device if selection changes
            if selected_device_name and selected_device_name in loopback_devices:
                # Update the recorder's target device immediately
                target_index = loopback_devices[selected_device_name]
                if system_recorder.device_index != target_index:
                    print(f"SWITCHING DEVICE TO: {selected_device_name} (Index: {target_index})")
                    system_recorder.set_device(target_index)
                    if system_recorder.is_running:
                        system_recorder.stop()
                        st.rerun() # Restart required to apply change
            
            # --- DIRECT AUDIO TEST (Main Thread) ---
            if st.button("🔊 Direct Audio Test (3s)"):
                st.info("Testing audio directly (bypassing background thread)...")
                try:
                    # Run a short blocking recording
                    test_pa = pyaudio.PyAudio()
                    test_idx = loopback_devices[selected_device_name]
                    test_dev = test_pa.get_device_info_by_index(test_idx)
                    
                    st.write(f"Opening stream on: {test_dev['name']}")
                    st.write(f"Rate: {test_dev['defaultSampleRate']}, Channels: {test_dev['maxInputChannels']}")
                    
                    stream = test_pa.open(
                        format=pyaudio.paFloat32,
                        channels=int(test_dev['maxInputChannels']),
                        rate=int(test_dev['defaultSampleRate']),
                        input=True,
                        input_device_index=test_idx,
                        frames_per_buffer=1024
                    )
                    
                    chunks = []
                    prog = st.progress(0)
                    for i in range(30): # ~3 seconds
                        data = stream.read(int(test_dev['defaultSampleRate'] / 10), exception_on_overflow=False)
                        chunks.append(np.frombuffer(data, dtype=np.float32))
                        prog.progress((i+1)/30)
                        
                    stream.stop_stream()
                    stream.close()
                    test_pa.terminate()
                    
                    full_data = np.concatenate(chunks)
                    rms = np.sqrt(np.mean(full_data**2))
                    st.write(f"**Result RMS:** {rms:.6f}")
                    
                    if rms > 0.0001:
                        st.success("✅ AUDIO DETECTED! The app can hear you.")
                    else:
                        st.error("❌ SILENCE. Windows is blocking audio. Disable Exclusive Mode.")
                        
                except Exception as e:
                    st.error(f"Test Failed: {e}")
            if selected_device_name:
                system_recorder.set_device(loopback_devices[selected_device_name])
        else:
            st.warning("No System Audio devices found.")

    st.markdown("### Audio Levels")
    # Moved to main column for better visibility
    st.caption("Check the visual meter in the main screen to verify audio.")
    
    st.markdown("---")
    st.subheader("🧪 Manual Test Mode")
    st.caption("Type a question to test the AI's response logic using your loaded Context (Resume/Script). Requires API Key.")
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
        
        # Validation & Feedback
        if "Error" in r_text:
            st.error(f"Resume Load Failed: {r_text}")
        elif len(r_text) < 50 and resume_file:
            st.warning("⚠️ Resume seems very short or empty. Is it an image-based PDF?")
            
        if "Error" in s_text_file:
            st.error(f"Script Load Failed: {s_text_file}")
            
        # Summary
        stats = []
        if r_text: stats.append(f"✅ Resume ({len(r_text.split())} words)")
        if job_desc: stats.append(f"✅ JD ({len(job_desc.split())} words)")
        if full_script: stats.append(f"✅ Script ({len(full_script.split())} words)")
        
        if stats:
            st.success(f"Context Loaded Successfully! \n\n" + " | ".join(stats))
        else:
            st.warning("Context is empty. Please upload files or paste text.")

    # --- HUD Settings ---
    st.markdown("---")
    with st.sidebar.expander("🎨 HUD / Overlay Settings", expanded=True):
        st.caption("Customize the Live Answer Box size and opacity.")
        
        # REMOVED X/Y Sliders - replaced by Drag & Drop JS
        # col_pos1, col_pos2 = st.columns(2)
        # with col_pos1:
        #     hud_y_pos = st.slider("Vertical (Top %)", 0, 100, 10, help="0% = Top, 100% = Bottom")
        # with col_pos2:
        #     hud_x_pos = st.slider("Horizontal (Left %)", 0, 100, 50, help="0% = Left, 100% = Right")
            
        col_size1, col_size2 = st.columns(2)
        with col_size1:
            hud_width = st.slider("Width (px)", 300, 1000, 500)
        with col_size2:
            hud_opacity = st.slider("Opacity", 0.1, 1.0, 0.95)
            
        hud_font_size = st.slider("Font Size (px)", 12, 24, 16)
        
        st.info("💡 **Tip:** You can now DRAG the box anywhere on the screen!")

    if st.button("Clear Transcript"):
        st.session_state["last_transcript"] = ""
        st.rerun()
        
    # --- Live Interview HUD (Custom CSS & JS) ---
    # Enhanced CSS for "Movable" look and "Always on Top"
    # JS handles dragging and persistence via localStorage
    
    # Generate unique ID for the box to attach events
    hud_id = "live-answer-hud"
    
    st.markdown(f"""
<style>
    #{hud_id} {{
        position: fixed;
        /* Default Position (Center-ish) if no localStorage found */
        top: 10%;
        left: 50%;
        transform: translate(-50%, 0); /* Center horizontally by default */
        
        width: {hud_width}px;
        max-height: 80vh;
        overflow-y: auto;
        background-color: rgba(20, 20, 20, {hud_opacity});
        color: #e0e0e0;
        padding: 0; /* Remove padding to handle drag header */
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        z-index: 999999; /* Super high z-index to overwrite everything */
        font-family: 'Segoe UI', sans-serif;
        font-size: {hud_font_size}px;
        backdrop-filter: blur(10px);
        transition: opacity 0.3s ease; /* Only animate opacity, not pos (interferes with drag) */
    }}
    
    .hud-drag-handle {{
        background: rgba(255, 255, 255, 0.1);
        color: #bbb;
        padding: 5px 10px;
        font-size: 0.8em;
        text-align: center;
        cursor: move;
        border-top-left-radius: 12px;
        border-top-right-radius: 12px;
        user-select: none;
        letter-spacing: 1px;
        font-weight: bold;
        text-transform: uppercase;
    }}
    
    .hud-content {{
        padding: 20px;
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
    #{hud_id}::-webkit-scrollbar {{
        width: 8px;
    }}
    #{hud_id}::-webkit-scrollbar-track {{
        background: rgba(255, 255, 255, 0.05);
    }}
    #{hud_id}::-webkit-scrollbar-thumb {{
        background: #555;
        border-radius: 4px;
    }}
    #{hud_id}::-webkit-scrollbar-thumb:hover {{
        background: #777;
    }}
</style>

<script>
(function() {{
    const hud = document.getElementById('{hud_id}');
    if (!hud) return;
    
    // --- 1. RESTORE POSITION ---
    const savedTop = localStorage.getItem('hud_top');
    const savedLeft = localStorage.getItem('hud_left');
    
    if (savedTop && savedLeft) {{
        hud.style.top = savedTop;
        hud.style.left = savedLeft;
        hud.style.transform = 'none'; // Remove centering transform if moved
    }}
    
    // --- 2. DRAG LOGIC ---
    const handle = hud.querySelector('.hud-drag-handle');
    let isDragging = false;
    let startX, startY, initialLeft, initialTop;
    
    handle.addEventListener('mousedown', (e) => {{
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        
        const rect = hud.getBoundingClientRect();
        initialLeft = rect.left;
        initialTop = rect.top;
        
        // Remove transform to allow absolute positioning to work predictably
        hud.style.transform = 'none';
        hud.style.left = initialLeft + 'px';
        hud.style.top = initialTop + 'px';
        
        handle.style.cursor = 'grabbing';
        document.body.style.userSelect = 'none'; // Prevent text selection
    }});
    
    document.addEventListener('mousemove', (e) => {{
        if (!isDragging) return;
        
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        
        hud.style.left = (initialLeft + dx) + 'px';
        hud.style.top = (initialTop + dy) + 'px';
    }});
    
    document.addEventListener('mouseup', () => {{
        if (!isDragging) return;
        isDragging = false;
        handle.style.cursor = 'move';
        document.body.style.userSelect = '';
        
        // --- 3. SAVE POSITION ---
        localStorage.setItem('hud_top', hud.style.top);
        localStorage.setItem('hud_left', hud.style.left);
    }});
    
}})();
</script>
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
        st.info("🎧 Listening to System Audio (Interviewer's Voice only).")
        st.caption("ℹ️ This captures what comes out of your **Speakers**. It does NOT hear your microphone. To test, play a YouTube video or start a call.")
        
        # Audio Meter UI
        st.markdown("**Audio Level:**")
        rms_placeholder = st.progress(0.0)
        status_placeholder = st.empty()
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Start Listening", type="primary"):
                try:
                    system_recorder.start()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("""
        **Troubleshooting:**
        1. Ensure "Exclusive Mode" is **OFF** in Windows Sound Settings (See below).
        2. Play a continuous sound (e.g., YouTube video) while scanning.
        3. If using Zoom/Teams, ensure their output is set to the same device.
        """)
        
        if st.button("🔍 Deep Scan for Audio Signal"):
            found_name = system_recorder.auto_scan_and_start()
            if found_name:
                st.session_state.selected_device_name = found_name # Update UI selection if possible
                st.rerun()

        with st.expander("ℹ️ How to fix 'Exclusive Mode' (Important!)"):
            st.markdown("""
            **Windows prevents recording if another app has Exclusive Control.**
            
            1. Open **Control Panel** -> **Sound**.
            2. Go to **Playback** tab.
            3. Right-click your default device (green checkmark) -> **Properties**.
            4. Go to **Advanced** tab.
            5. **UNCHECK**: "Allow applications to take exclusive control of this device".
            6. Click **Apply/OK**.
            7. **Restart this app**.
            """)
        
        if st.button("Stop Listening"):
            system_recorder.stop()
            st.rerun()
            
        if st.button("🚨 FORCE TRANSCRIBE NOW (Debug)", help="Bypasses silence detection and sends whatever is in buffer"):
             # This is a hack to force the buffer to flush
             # We can't easily access the local variable 'speech_buffer' from here as it's inside the 'if is_playing' block below
             # But we can set a flag in session state that the loop checks?
             st.session_state["force_transcribe"] = True
             
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
<div id="{hud_id}" class="floating-answer-box">
<div class="hud-drag-handle">:: Drag Me ::</div>
<div class="hud-content">
<div class="transcript-box">
<span class="label">Status:</span> {last_text} {processing_status}
</div>
<div class="answer-box">
<h4>Live Answer:</h4>
<p>{current_answer}</p>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# Main Loop for processing
if is_playing:
    # Check if client is available
    if not client:
        st.warning("⚠️ No API Key detected! Audio meter will work, but AI will NOT respond. Please enter API Key in Sidebar.")

    # Buffering Logic
    speech_buffer = []
    silence_frames = 0
    MIN_CHUNKS_TO_PROCESS = 3 # 0.3 seconds (Even faster)
    MAX_BUFFER_SIZE = 150 # 15 seconds
    SILENCE_THRESHOLD = silence_threshold_val # USE SLIDER VALUE
    SILENCE_DURATION_TRIGGER = 5 # Increased to 0.5s to avoid cutting off mid-word
    
    # State tracking for UI
    is_speaking = False
    debug_placeholder = st.empty()

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
                
                # Boost it for visibility (Auto-Gain Visualization)
                display_level = min(rms * 500, 1.0) 
                rms_placeholder.progress(float(display_level))
                
                # --- DYNAMIC STATUS UPDATE ---
                status_color = "🟢" if rms > SILENCE_THRESHOLD else "⚪"
                status_msg = "LISTENING" if rms > SILENCE_THRESHOLD else "WAITING"
                status_placeholder.markdown(f"**RMS:** `{rms:.6f}` | **Threshold:** `{SILENCE_THRESHOLD}` | {status_color} {status_msg}")
                
                # --- UPDATE HUD REAL-TIME ---
                # We update the 'processing_status' part of the HUD
                if rms > SILENCE_THRESHOLD:
                    new_status_text = f"Listening... (RMS: {rms:.4f})"
                else:
                    new_status_text = "Waiting..."
                    
                suggestion_placeholder.markdown(f"""
<div id="{hud_id}" class="floating-answer-box">
<div class="hud-drag-handle">:: Drag Me ::</div>
<div class="hud-content">
<div class="transcript-box">
<span class="label">Status:</span> {new_status_text}
</div>
<div class="answer-box">
<h4>Live Answer:</h4>
<p>{current_answer}</p>
</div>
</div>
</div>
""", unsafe_allow_html=True)
                
            # --- BUFFERING & SILENCE DETECTION ---
            speech_buffer.append(audio_data)
            
            # CHECK MAX BUFFER SIZE (Force Send)
            if len(speech_buffer) > MAX_BUFFER_SIZE:
                 # print("MAX BUFFER REACHED - FORCING PROCESS")
                 # Pretend we hit silence to force processing
                 rms = 0.0
                 silence_frames = SILENCE_DURATION_TRIGGER + 10
            
            # CHECK FORCE FLAG
            if st.session_state.get("force_transcribe", False):
                 print("FORCE TRANSCRIBE TRIGGERED")
                 st.session_state["force_transcribe"] = False # Reset
                 # Pretend we have silence to trigger flush
                 rms = 0.0 
                 silence_frames = SILENCE_DURATION_TRIGGER + 1
                 # Also ensure we have enough data?
                 if len(speech_buffer) < MIN_CHUNKS_TO_PROCESS:
                     # Pad with silence if needed just to make it run
                     speech_buffer.extend([np.zeros_like(audio_data)] * 5)
            
            # --- HUD AUDIO METER ---
            # Show a visual indicator in the HUD so user knows if audio is being captured
            if int(time.time() * 10) % 2 == 0: # Update every ~200ms to avoid flickering
                mic_status_icon = "🟢" if rms > SILENCE_THRESHOLD else "⚪"
                mic_level_bar = "I" * int(rms * 100) # Simple text bar
                if len(mic_level_bar) > 10: mic_level_bar = mic_level_bar[:10]
                
                # Check for "DEAD AUDIO" (No capture for 5+ seconds)
                if rms == 0.0:
                    st.session_state["zero_audio_frames"] = st.session_state.get("zero_audio_frames", 0) + 1
                else:
                    st.session_state["zero_audio_frames"] = 0
                
                # If 5 seconds of PURE ZERO (not just silence, but 0.0 data), warn user
                if st.session_state.get("zero_audio_frames", 0) > 50: # 50 * 0.1s = 5s
                    warning_html = """
<div style="background: #ff4444; color: white; padding: 5px; border-radius: 4px; font-size: 0.8em; margin-bottom: 5px;">
    ⚠️ NO AUDIO DETECTED! <br/>
    Check 'Audio Source' in Sidebar. Try a different speaker.
</div>
"""
                else:
                    warning_html = ""

                # Update HUD status line
                suggestion_placeholder.markdown(f"""
<div id="{hud_id}" class="floating-answer-box">
<div class="hud-drag-handle">:: Drag Me ::</div>
<div class="hud-content">
<div class="transcript-box">
{warning_html}
<span class="label">Status:</span> {mic_status_icon} Listening... <span style="color:#4CAF50; font-weight:bold;">{mic_level_bar}</span>
</div>
<div class="answer-box">
<h4>Live Answer:</h4>
<p>{current_answer}</p>
</div>
</div>
</div>
""", unsafe_allow_html=True)

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
                     # Buffer is all silence, clear it to avoid processing empty noise later
                     speech_buffer = []
                     silence_frames = 0
                continue

            # --- PROCESS BUFFER ---
            # Concatenate all chunks
            full_audio = np.concatenate(speech_buffer)
            
            # Check average RMS of the phrase to ensure it's not just background hum
            phrase_rms = np.sqrt(np.mean(full_audio**2))
            
            # Reset Buffer
            speech_buffer = []
            silence_frames = 0
            
            # RMS Threshold to avoid processing empty noise (e.g., HVAC, distant sounds)
            # Reduced to 0.001 to ensure we don't miss quiet questions
            if phrase_rms < 0.001:
                # print(f"Skipping noise (RMS: {phrase_rms:.4f})")
                continue
            
            # CHECK CLIENT BEFORE PROCESSING
            if not client:
                st.toast("⚠️ Audio detected, but NO API KEY. Cannot transcribe.", icon="🛑")
                continue

            # Show "Voice Detected" temporarily
            suggestion_placeholder.markdown(f"""
<div class="floating-answer-box">
<div class="transcript-box">
<span class="label">Status:</span> 🎤 Voice Detected (RMS: {phrase_rms:.3f})... Processing...
</div>
<div class="answer-box">
<h4>Live Answer:</h4>
<p>{current_answer}</p>
</div>
</div>
""", unsafe_allow_html=True)

            # Process
            text = process_audio_chunk(full_audio, rate, client)
            
            if text is None:
                st.toast("⚠️ Transcription Failed. Check API Key or Console for details.", icon="❌")
                continue

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
                
                # Generate AI Response with STREAMING
                context = st.session_state.get('context_text', "No context loaded.")
                ai_answer = generate_ai_response(text, context, client, model_choice, placeholder=suggestion_placeholder)
                
                if ai_answer:
                    if "API KEY ERROR" in ai_answer:
                        st.error(ai_answer)
                    
                    st.session_state.ai_answer = ai_answer
                    # Update HUD with the Final Answer (redundant if streamed, but ensures clean state)
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
