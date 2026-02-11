
import os
import sys
import time
import numpy as np
try:
    import pyaudiowpatch as pyaudio
except ImportError:
    import pyaudio
import wave
import scipy.signal
from openai import OpenAI

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
BASE_URL = None
TRANSCRIPTION_MODEL = "whisper-1"

print("\n⚙️  Configuration Check:")
if GROQ_API_KEY:
    print("   ✅ Found GROQ_API_KEY. Using Groq (whisper-large-v3).")
    API_KEY = GROQ_API_KEY
    BASE_URL = "https://api.groq.com/openai/v1"
    TRANSCRIPTION_MODEL = "whisper-large-v3"
elif API_KEY:
    print("   ✅ Found OPENAI_API_KEY. Using OpenAI (whisper-1).")
else:
    print("   ⚠️  No API Key found in environment.")
    print("   [1] OpenAI")
    print("   [2] Groq")
    choice = input("   Select Provider (1/2): ").strip()
    
    API_KEY = input("   Enter API Key: ").strip()
    if choice == "2":
        BASE_URL = "https://api.groq.com/openai/v1"
        TRANSCRIPTION_MODEL = "whisper-large-v3"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def record_audio(duration=5, filename="debug_audio.wav"):
    print(f"\n🎤  Recording {duration} seconds of System Audio...")
    print("    (Please play some YouTube/Music NOW)")
    
    p = pyaudio.PyAudio()
    
    # 1. Find Loopback Device
    loopback_index = None
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if "stereo mix" in dev['name'].lower() or "loopback" in dev['name'].lower():
            loopback_index = i
            print(f"    Found Loopback Device: {dev['name']} (Index {i})")
            break
            
    # Fallback to default input if no explicit loopback found
    if loopback_index is None:
        def_info = p.get_default_input_device_info()
        loopback_index = def_info['index']
        print(f"    Using Default Input: {def_info['name']} (Index {loopback_index})")

    # 2. Record
    CHUNK = 1024
    FORMAT = pyaudio.paInt16 # Use Int16 for standard WAV compatibility
    CHANNELS = 1
    RATE = 16000 # Whisper native rate
    
    frames = []
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=loopback_index,
                        frames_per_buffer=CHUNK)
        
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
            
            # Simple visual meter
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data**2))
            bars = "|" * int(rms / 500)
            sys.stdout.write(f"\r    Level: {bars[:50]:<50}")
            sys.stdout.flush()
            
        print("\n    ✅ Recording Complete.")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # 3. Save WAV
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"    💾 Saved to {filename}")
        return True
        
    except Exception as e:
        print(f"\n    ❌ Recording Failed: {e}")
        return False

def test_transcription(filename="debug_audio.wav"):
    print(f"\n📝  Testing Transcription on {filename}...")
    
    if not os.path.exists(filename):
        print("    ❌ File not found.")
        return

    try:
        with open(filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=audio_file,
                language="en"
            )
            print(f"    ✅ Transcription Success: '{transcript.text}'")
            return True
    except Exception as e:
        print(f"    ❌ Transcription Failed: {e}")
        return False

if __name__ == "__main__":
    print("=== INTERVIEW AI DIAGNOSTIC TOOL ===")
    if record_audio():
        test_transcription()
    print("\n=== DIAGNOSTIC COMPLETE ===")
    input("Press Enter to exit...")
