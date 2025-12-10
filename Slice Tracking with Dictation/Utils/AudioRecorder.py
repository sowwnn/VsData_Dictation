import os
import datetime
import logging
import csv
import threading
import time
import subprocess
import sys
import sounddevice as sd
import numpy as np
import wave

class AudioRecorder:
    """Audio recorder using sounddevice for Windows API access."""
    
    def __init__(self):
        self.isRecording = False
        self.outputFile = None
        self.startTime = None
        self.dictationMarkers = []  # List to store timestamps when dictation occurs
        self.inputDevice = None
        self.process = None
        self.recording_thread = None
        self.audio_data = []
        self.sample_rate = 16000
        self.channels = 1
        
    def setInputDevice(self, deviceIndex):
        """Set the input device to use for recording."""
        self.inputDevice = deviceIndex
        logging.info(f"Set audio input device to index: {deviceIndex}")
        
    def startRecording(self, outputFile=None):
        """Start audio recording using sounddevice."""
        if self.isRecording:
            return False
            
        self.outputFile = outputFile
        if self.outputFile:
            # Create output directory if needed
            outputDir = os.path.dirname(self.outputFile)
            if outputDir and not os.path.exists(outputDir):
                try:
                    os.makedirs(outputDir)
                except Exception as e:
                    logging.error(f"Error creating output directory: {str(e)}")
        
        # Clear markers list and audio data
        self.dictationMarkers = []
        self.audio_data = []
        
        # Record start time
        self.startTime = datetime.datetime.now(datetime.timezone.utc)
        
        try:
            # Start recording in a separate thread
            self.isRecording = True
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.start()
            logging.info("Started audio recording")
            return True
        except Exception as e:
            logging.error(f"Error starting recording: {str(e)}")
            self.isRecording = False
            return False
    
    def _record_audio(self):
        """Record audio using sounddevice."""
        try:
            # Open audio stream
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.int16,
                device=self.inputDevice if self.inputDevice else None
            )
            
            stream.start()
            
            # Record until stopped
            while self.isRecording:
                data, overflowed = stream.read(self.sample_rate)
                if not overflowed:
                    self.audio_data.append(data)
            
            # Stop and close stream
            stream.stop()
            stream.close()
            
            # Save audio data to file
            if self.outputFile and self.audio_data:
                self._save_audio_file()
                
        except Exception as e:
            logging.error(f"Error during recording: {str(e)}")
            self.isRecording = False
    
    def _save_audio_file(self):
        """Save recorded audio data to WAV file."""
        try:
            # Combine all audio data
            audio_data = np.concatenate(self.audio_data, axis=0)
            
            # Save as WAV file
            with wave.open(self.outputFile, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            logging.info(f"Audio saved to {self.outputFile}")
        except Exception as e:
            logging.error(f"Error saving audio file: {str(e)}")
    
    def stopRecording(self):
        """Stop audio recording."""
        if not self.isRecording:
            return False
            
        # Set the flag to stop recording
        self.isRecording = False
        
        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join()
            self.recording_thread = None
        
        logging.info("Recording stopped")
        return True
    
    def markDictation(self):
        """Mark the current time as having dictation."""
        if self.isRecording and self.startTime:
            currentTime = datetime.datetime.now(datetime.timezone.utc)
            elapsed = (currentTime - self.startTime).total_seconds()
            self.dictationMarkers.append(elapsed)
            logging.info(f"Dictation marker added at {elapsed:.2f} seconds")
            return elapsed
        return None
    
    def getDictationMarkers(self):
        """Get the list of dictation markers (timestamps)."""
        return self.dictationMarkers
        
    def saveDictationMarkers(self, outputFile):
        """Save dictation markers to a CSV file."""
        if not self.dictationMarkers or not outputFile:
            return False
            
        try:
            with open(outputFile, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['TimeSec'])
                for timestamp in self.dictationMarkers:
                    writer.writerow([f"{timestamp:.3f}"])
            logging.info(f"Saved {len(self.dictationMarkers)} dictation markers to {outputFile}")
            return True
        except Exception as e:
            logging.error(f"Error saving dictation markers: {str(e)}")
            return False