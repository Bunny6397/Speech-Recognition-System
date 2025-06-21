# Speech-Recognition-System
Here's a basic Speech Recognition System built using Python and libraries like SpeechRecognition and pydub for handling audio, and optionally Wav2Vec2 from HuggingFace for deep learning-based transcription.

## Overview
This project implements a basic speech-to-text system using Python. It supports:
- Google Web Speech API via `SpeechRecognition`
- Facebook’s Wav2Vec2 model via HuggingFace for deep learning transcription

### Run with SpeechRecognition:
```bash
python main.py
###python code
import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Load audio file
audio_path = "sample_audio.wav"
with sr.AudioFile(audio_path) as source:
    audio_data = recognizer.record(source)
    print("Recognizing...")

    try:
        # Convert speech to text using Google Web Speech API
        text = recognizer.recognize_google(audio_data)
        print("Transcription:", text)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError:
        print("Request error from API")
