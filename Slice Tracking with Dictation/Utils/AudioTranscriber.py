import os
import logging
import json
import traceback
import wave
from google.cloud import speech
from google.oauth2 import service_account

class AudioTranscriber:
    def __init__(self, service="google", credentials_path=None, language_code="en-US", model=None):
        self.service = service.lower()
        self.transcription_info = {}
        self.client = None
        self.language_code = language_code
        self.model = model

        # The 'medical_dictation' model only works with 'en-US'.
        if self.model == "medical_dictation" and self.language_code != "en-US":
            logging.warning(
                f"The 'medical_dictation' model only supports 'en-US'. "
                f"Ignoring model for language '{self.language_code}'."
            )
            self.model = None

        if self.service == "google":
            self.initialize_google_speech(credentials_path)
        else:
            logging.error(f"Unsupported service: {service}")

    def initialize_google_speech(self, credentials_path=None):
        """
        Initializes the Google Cloud Speech client.
        It can use a specific credentials file or fall back to environment variables.
        """
        try:
            # Priority 1: Use the provided credentials file path
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.client = speech.SpeechClient(credentials=credentials)
                logging.info(f"Google Cloud Speech client initialized using file: {credentials_path}")
            # Priority 2: Fallback to environment variables if no path is provided
            else:
                self.client = speech.SpeechClient()
                logging.info("Google Cloud Speech client initialized using default credentials (environment variables).")
                if not credentials_path:
                     logging.warning("No credentials_path provided. Ensure GOOGLE_APPLICATION_CREDENTIALS is set.")
                else:
                     logging.error(f"Provided credentials_path does not exist: {credentials_path}")
                     self.client = None
                     return

        except Exception as e:
            logging.error(f"Failed to initialize Google Cloud Speech client: {e}")
            logging.error("Please ensure you provide a valid service account JSON file or set the GOOGLE_APPLICATION_CREDENTIALS environment variable.")
            self.client = None

    def transcribe_audio(self, audio_file_path):
        """
        Transcribes the given audio file using Google Cloud Speech-to-Text.
        """
        if not self.client:
            logging.error("Google Cloud Speech client is not initialized.")
            return False

        if not os.path.exists(audio_file_path):
            logging.error(f"Audio file not found: {audio_file_path}")
            return False
            
        if os.path.getsize(audio_file_path) < 100: # Google API requires non-empty content
            logging.error(f"Audio file is too small or empty: {audio_file_path}")
            return False

        try:
            logging.info(f"Starting transcription for {audio_file_path} with Google Speech-to-Text...")

            with open(audio_file_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            
            # --- Dynamically detect sample rate ---
            try:
                with wave.open(audio_file_path, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                logging.info(f"Automatically detected sample rate: {sample_rate} Hz")
            except Exception as e:
                logging.warning(f"Could not detect sample rate from WAV header: {e}. Falling back to a common rate.")
                # If reading header fails, we can fall back, but this is less ideal.
                # 16000 is common for voice recording, which is likely our case.
                sample_rate = 16000

            config_dict = {
                "encoding": speech.RecognitionConfig.AudioEncoding.LINEAR16,
                "sample_rate_hertz": sample_rate,  # Use the detected rate
                "language_code": self.language_code,
                "enable_word_time_offsets": True,
                "enable_automatic_punctuation": True,
            }
            # Conditionally add the model to the config if it's specified
            if self.model:
                config_dict["model"] = self.model

            config = speech.RecognitionConfig(**config_dict)

            response = self.client.recognize(config=config, audio=audio)

            segments = []
            full_text_parts = []
            # Each result in the response is a segment of speech separated by a long pause.
            for result in response.results:
                if not result.alternatives or not result.alternatives[0].words:
                    continue

                alternative = result.alternatives[0]
                full_text_parts.append(alternative.transcript.strip())
                
                # The start time of the first word and end time of the last word in this result.
                start_time = alternative.words[0].start_time.total_seconds()
                end_time = alternative.words[-1].end_time.total_seconds()
                
                segments.append({
                    "start_time": round(start_time, 2),
                    "end_time": round(end_time, 2),
                    "text": alternative.transcript.strip()
                })

            full_text = " ".join(full_text_parts)

            self.transcription_info = {
                "language": self.language_code,
                "language_probability": 1.0,
                "duration_seconds": segments[-1]['end_time'] if segments else 0,
                "segments": segments,
                "full_text": full_text
            }
            
            print(f"[Transcriber] Transcription completed with model: {self.model or 'default'}. Found {len(segments)} segments.")
            return True

        except Exception as e:
            logging.error(f"Error during Google Speech-to-Text transcription: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def save_transcription(self, output_file):
        """Saves the transcription data to a JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.transcription_info, f, indent=2, ensure_ascii=False)
            logging.info(f"Transcription saved to {output_file}")
            return True
        except Exception as e:
            logging.error(f"Error saving transcription: {str(e)}")
            return False

    def get_transcription(self):
        """Returns the stored transcription information."""
        return self.transcription_info

if __name__ == "__main__":
    # This is a placeholder for testing.
    # You would need a sample audio file and Google Cloud authentication.
    # Example usage:
    # ----------------
    # if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    #     print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
    #     print("Please set it to the path of your Google Cloud service account key file.")
    # else:
    #     transcriber = AudioTranscriber(service="google")
    #     # Make sure 'test_audio.wav' exists and is a valid WAV file
    #     if transcriber.transcribe_audio("test_audio.wav"):
    #         print("Transcription successful:")
    #         print(json.dumps(transcriber.get_transcription(), indent=2))
    #         transcriber.save_transcription("transcription_google.json")
    pass