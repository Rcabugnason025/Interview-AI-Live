
import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()

print("--- SEARCHING FOR LOOPBACK DEVICES ---")
try:
    # Get default WASAPI info
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    print(f"DEFAULT OUTPUT DEVICE: {default_speakers['name']} (Index: {default_speakers['index']})")
    
    print("\n--- ALL LOOPBACK CANDIDATES ---")
    for loopback in p.get_loopback_device_info_generator():
        print(f"Device: {loopback['name']}")
        print(f"  Index: {loopback['index']}")
        print(f"  Rate: {loopback['defaultSampleRate']}")
        print(f"  Channels: {loopback['maxInputChannels']}")
        print("---")
        
except Exception as e:
    print(f"Error: {e}")

p.terminate()
