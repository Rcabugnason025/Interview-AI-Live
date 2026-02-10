import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()

print("Searching for WASAPI Loopback devices...")
try:
    # Get default WASAPI speakers
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    
    print(f"Default Output Device: {default_speakers['name']}")
    print(f"  Sample Rate: {default_speakers['defaultSampleRate']}")
    print(f"  Max Output Channels: {default_speakers['maxOutputChannels']}")
    
    target_device = default_speakers
    
    if not default_speakers["isLoopbackDevice"]:
        print("  (Not a loopback device, searching for loopback counterpart...)")
        found = False
        for loopback in p.get_loopback_device_info_generator():
            print(f"  Found Loopback Candidate: {loopback['name']} (Index: {loopback['index']})")
            if loopback["name"] == default_speakers["name"]:
                target_device = loopback
                found = True
                print("  -> MATCH FOUND!")
                break
        if not found:
            print("  -> No matching loopback found. Using first available loopback.")
            try:
                target_device = next(p.get_loopback_device_info_generator())
                print(f"  -> Selected: {target_device['name']}")
            except StopIteration:
                print("  -> No loopback devices found at all!")
                exit()
    
    # Check simple matching logic
    if default_speakers["name"] in target_device["name"]:
        print("  -> Simple String Match: YES")
    else:
        print("  -> Simple String Match: NO (This explains why auto-select failed)")

    print(f"Attempting to open: {target_device['name']} (Index: {target_device['index']})")
    
    # Try different configurations
    rates = [int(target_device["defaultSampleRate"]), 44100, 48000]
    channels_list = [int(target_device["maxInputChannels"]), 2, 1]
    
    for r in list(set(rates)):
        for c in list(set(channels_list)):
            if c <= 0: continue
            print(f"Testing Rate: {r}, Channels: {c}...", end="")
            try:
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=c,
                    rate=r,
                    input=True,
                    input_device_index=target_device["index"],
                    frames_per_buffer=1024
                )
                print(" SUCCESS!")
                stream.close()
            except Exception as e:
                print(f" FAILED ({e})")

except Exception as e:
    print(f"General Error: {e}")

p.terminate()
