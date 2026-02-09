import pyaudiowpatch as pyaudio
import sys

print(f"Python: {sys.version}")
p = pyaudio.PyAudio()

print("\n--- WASAPI Loopback Devices ---")
try:
    # Get WASAPI Host API
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    print(f"WASAPI Host API Index: {wasapi_info['index']}")
    
    default_output = wasapi_info["defaultOutputDevice"]
    print(f"Default Output Device Index: {default_output}")
    
    default_dev_info = p.get_device_info_by_index(default_output)
    print(f"Default Output Device Name: {default_dev_info['name']}")
    
    # List all loopback devices
    found_loopback = False
    for loopback in p.get_loopback_device_info_generator():
        print(f"\nLoopback Device Found:")
        print(f"  Index: {loopback['index']}")
        print(f"  Name: {loopback['name']}")
        print(f"  Sample Rate: {loopback['defaultSampleRate']}")
        print(f"  Input Channels: {loopback['maxInputChannels']}")
        print(f"  Output Channels: {loopback['maxOutputChannels']}")
        found_loopback = True
        
        # Try to open a stream
        print("  > Attempting to open stream...")
        try:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=int(loopback["maxInputChannels"]),
                rate=int(loopback["defaultSampleRate"]),
                input=True,
                input_device_index=loopback["index"],
                frames_per_buffer=1024
            )
            print("  > SUCCESS! Stream opened with paFloat32.")
            stream.close()
        except Exception as e:
            print(f"  > Failed with paFloat32: {e}")
            
            try:
                print("  > Retrying with paInt16...")
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=int(loopback["maxInputChannels"]),
                    rate=int(loopback["defaultSampleRate"]),
                    input=True,
                    input_device_index=loopback["index"],
                    frames_per_buffer=1024
                )
                print("  > SUCCESS! Stream opened with paInt16.")
                stream.close()
            except Exception as e2:
                print(f"  > Failed with paInt16: {e2}")

    if not found_loopback:
        print("\nNo loopback devices found via generator.")

except Exception as e:
    print(f"Error: {e}")

p.terminate()
