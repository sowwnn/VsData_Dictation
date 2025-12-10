#!/usr/bin/env python3
"""
Audio recorder script to be run as a separate process.
This script records audio and saves it to a WAV file.
"""

import os
import sys
import json
import signal
import time
import wave
import threading

# Try to import audio libraries with error handling
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    print("PyAudio module not available. Audio recording will be disabled.")
    AUDIO_AVAILABLE = False
    sys.exit(1)

# Check command line arguments
if len(sys.argv) < 3:
    print("Usage: audio_recorder_script.py <output_file> <markers_file> [device_index]")
    sys.exit(1)

output_file = sys.argv[1]
markers_file = sys.argv[2]

# Optional device index
device_index = None
if len(sys.argv) > 3:
    try:
        device_index = int(sys.argv[3])
    except ValueError:
        print(f"Warning: Invalid device index {sys.argv[3]}, using default")

# Create directory if it doesn't exist
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        sys.exit(1)

# Initialize variables
frames = []
is_recording = True
dictation_markers = []

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Load any existing markers
if os.path.exists(markers_file):
    try:
        with open(markers_file, 'r') as f:
            dictation_markers = json.load(f)
    except Exception as e:
        print(f"Error reading markers file: {str(e)}")

# Set up signal handler for graceful termination
def signal_handler(sig, frame):
    global is_recording
    print("Received termination signal, stopping recording...")
    is_recording = False

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Function to periodically check for new dictation markers
def check_markers():
    global dictation_markers
    while is_recording:
        try:
            if os.path.exists(markers_file):
                with open(markers_file, 'r') as f:
                    new_markers = json.load(f)
                    if new_markers != dictation_markers:
                        dictation_markers = new_markers
                        print(f"Updated dictation markers: {len(dictation_markers)} markers")
        except Exception as e:
            print(f"Error reading markers file: {str(e)}")
        
        time.sleep(1)  # Check every second

# Start markers checking thread
markers_thread = threading.Thread(target=check_markers)
markers_thread.daemon = True
markers_thread.start()

# Initialize PyAudio
p = pyaudio.PyAudio()

try:
    # Open stream
    kwargs = {
        'format': FORMAT,
        'channels': CHANNELS,
        'rate': RATE,
        'input': True,
        'frames_per_buffer': CHUNK
    }
    
    # Add device index if specified
    if device_index is not None:
        kwargs['input_device_index'] = device_index
    
    stream = p.open(**kwargs)
    
    print(f"Recording started to {output_file}")
    print(f"Press Ctrl+C to stop recording")
    
    # Recording loop
    while is_recording:
        try:
            data = stream.read(CHUNK)
            frames.append(data)
        except Exception as e:
            print(f"Error reading audio data: {str(e)}")
            break
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    
    # Save the recorded data as a WAV file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    print(f"Recording saved to {output_file}")
    
    # Save markers to the temp file (if we have any)
    if dictation_markers:
        try:
            with open(markers_file, 'w') as f:
                json.dump(dictation_markers, f)
            print(f"Saved {len(dictation_markers)} dictation markers")
        except Exception as e:
            print(f"Error saving dictation markers: {str(e)}")
    
except Exception as e:
    print(f"Error in recording: {str(e)}")
finally:
    # Terminate PyAudio
    p.terminate()
    print("Recording process terminated")