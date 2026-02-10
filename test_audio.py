import sounddevice as sd
import sys
import os

print(f"Python: {sys.version}")
print(f"SoundDevice Version: {sd.__version__}")
print(f"SoundDevice File: {sd.__file__}")
try:
    print("Host APIs:")
    print(sd.query_hostapis())
    print("\nDevices:")
    print(sd.query_devices())
except Exception as e:
    print(f"Error querying devices: {e}")

try:
    print("\nChecking WasapiSettings:")
    print(sd.WasapiSettings(loopback=True))
except Exception as e:
    print(f"WasapiSettings error: {e}")
