"""Run this to diagnose microphone issues before starting the bot."""
import numpy as np
import sounddevice as sd

devices = sd.query_devices()
print("=== Scanning all input devices for signal (speak while it records) ===\n")

for i, dev in enumerate(devices):
    if dev["max_input_channels"] < 1:
        continue
    try:
        audio = sd.rec(
            int(2 * 16000), samplerate=16000, channels=1,
            dtype="float32", device=i
        )
        sd.wait()
        peak = float(np.max(np.abs(audio)))
        flag = "  <-- HAS SIGNAL" if peak > 0.001 else ""
        print(f"  [{i:2d}] peak={peak:.6f}{flag}  {dev['name']}")
    except Exception as e:
        print(f"  [{i:2d}] ERROR: {e}  {dev['name']}")

print("\nSet AUDIO_DEVICE to the index of the input device with signal.")
print()
print("=== Scanning output devices (will play a 440 Hz tone for 0.5 s each) ===\n")
import time
for i, dev in enumerate(devices):
    if dev["max_output_channels"] < 1:
        continue
    try:
        tone = (np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(0.5 * 44100))) * 0.3).astype(np.float32)
        sd.play(tone, samplerate=44100, device=i)
        sd.wait()
        print(f"  [{i:2d}] (played tone)  {dev['name']}")
        time.sleep(0.2)
    except Exception as e:
        print(f"  [{i:2d}] ERROR: {e}  {dev['name']}")

print("\nSet AUDIO_OUTPUT_DEVICE to the index of the device where you heard the tone.")
