import pyaudiowpatch as pyaudio
import numpy as np
import time

def verify():
    p = pyaudio.PyAudio()
    
    print("--- VERIFYING AUDIO SIGNAL ---")
    
    try:
        # Get default WASAPI info
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        print(f"Target Default Output: {default_speakers['name']}")
        
        # Find Loopback
        target_device = None
        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    target_device = loopback
                    break
            if not target_device:
                print("No matching loopback found. Picking first available.")
                target_device = next(p.get_loopback_device_info_generator())
        else:
            target_device = default_speakers
            
        print(f"Selected Loopback Device: {target_device['name']} (Index: {target_device['index']})")
        
        # Open Stream
        rate = int(target_device["defaultSampleRate"])
        channels = int(target_device["maxInputChannels"])
        
        print(f"Opening Stream (Rate: {rate}, Channels: {channels})...")
        
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=target_device["index"],
            frames_per_buffer=1024
        )
        
        print("Recording 3 seconds... (PLAY SOME AUDIO NOW!)")
        
        chunks = []
        start_time = time.time()
        while time.time() - start_time < 3:
            data = stream.read(1024, exception_on_overflow=False)
            chunks.append(np.frombuffer(data, dtype=np.float32))
            
        stream.stop_stream()
        stream.close()
        
        full_data = np.concatenate(chunks)
        rms = np.sqrt(np.mean(full_data**2))
        
        print(f"--- RESULT ---")
        print(f"Total Samples: {len(full_data)}")
        print(f"RMS Level: {rms:.6f}")
        
        if rms > 0.0001:
            print("✅ SUCCESS: Audio signal detected!")
        else:
            print("❌ FAILURE: Silence detected (RMS near 0).")
            print("Possible causes: 1. Exclusive Mode ON. 2. Volume Muted. 3. No audio playing.")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        
    p.terminate()

if __name__ == "__main__":
    verify()
