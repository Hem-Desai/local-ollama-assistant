import os
import sounddevice as sd
import whisper
import requests
import json
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, BarkModel
import numpy as np
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# Check available audio devices
def list_audio_devices():
    print("Available audio devices:")
    print(sd.query_devices())

# Record audio function that saves to C:\temp
def record_audio(filename, duration=5, fs=16000, device=None):
    print("Recording started...")
    
    try:
        # Ensure the directory C:\temp exists
        directory = r"C:\temp"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Set the full file path
        filepath = os.path.join(directory, filename + ".wav")

        # Record audio from the specified device (or default if None)
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16', device=device)
        print("Recording in progress...")
        sd.wait()  # Wait until recording is finished
        
        # Save the recording as a .wav file
        wavfile.write(filepath, fs, audio)
        print(f"Recording complete. File saved as {filepath}.")
        return filepath
    except Exception as e:
        print(f"Error during recording: {e}")
        return None

# Transcribe audio using Whisper
def transcribe_audio(filepath):
    print("Starting transcription...")
    try:
        # Check if file exists before transcription
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found!")
        
        model = whisper.load_model("base")
        result = model.transcribe(filepath)  # Directly use the WAV file
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

# Function to query the Ollama server (LLM query)
def query_ollama(model, prompt):
    print("Querying Ollama...")
    # Set the correct URL and headers
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }

    # Data payload with the dynamic prompt (from transcribed text)
    data = {
        "model": model,
        "prompt": prompt,  # Use the transcribed text as the prompt
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # Check if the request was successful
        if response.status_code == 200:
            response_text = response.text

            # Try to parse the response as JSON
            try:
                data = json.loads(response_text)
                actual_response = data["response"]
                return actual_response  # Return the actual response from Ollama
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print("Full response text:", response_text)  # Print raw response if parsing fails
        else:
            print(f"Error querying Ollama: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return None


# Main script to record, transcribe, query Ollama, and convert to TTS
if __name__ == "__main__":

    device = None  # Set the device ID if necessary (e.g., device=1)
    
    # Record audio, transcribe, and get response from Ollama
    audio_file = record_audio("user_audio", device=device)
    
    if audio_file:
        # Ensure transcription works by testing the file
        print("Testing file access for transcription...")
        if os.path.exists(audio_file):
            print(f"File {audio_file} exists and is ready for transcription.")
        else:
            print(f"File {audio_file} does not exist. Aborting transcription.")

        # Transcribe the audio
        text = transcribe_audio(audio_file)
        if text:
            print("You said:", text)
            # Query Ollama using the transcribed text
            response = query_ollama("llama2", text)
            if response:
                print("Assistant:", response)
                
                
