
import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()

print("\n--- FULL DEVICE LIST (WASAPI) ---")
try:
    # Iterate through ALL devices, not just loopback
    info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    numdevices = info.get('deviceCount')
    
    for i in range(0, numdevices):
        dev = p.get_device_info_by_host_api_device_index(info['index'], i)
        
        # Check if it's an output device (speakers/headphones)
        if dev['maxOutputChannels'] > 0:
            print(f"OUTPUT: {dev['name']} (Index: {dev['index']})")
            print(f"        IsLoopback: {dev['isLoopbackDevice']}")
            print(f"        SampleRate: {dev['defaultSampleRate']}")
            
        # Check if it's an input device (mic/line in)
        if dev['maxInputChannels'] > 0:
            print(f"INPUT:  {dev['name']} (Index: {dev['index']})")
            print(f"        IsLoopback: {dev['isLoopbackDevice']}")

except Exception as e:
    print(f"Error: {e}")

p.terminate()
