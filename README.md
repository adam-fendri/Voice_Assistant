# Voice Assistant Project

This project is a simple voice assistant application that uses Speech-to-Text (STT), a Large Language Model (LLM), and Text-to-Speech (TTS) to create a conversational AI. The assistant listens to audio input from a microphone, converts it to text, processes the text using a language model, and then converts the response back to speech.


## Introduction

The goal of this project is to demonstrate how a conversational AI can be built using a combination of various models:
- **Speech-to-Text (STT)**: Converts spoken language into text.
- **Large Language Model (LLM)**: Generates responses based on the input text.
- **Text-to-Speech (TTS)**: Converts the response text back into spoken language.

### Components

1. **Speech to Text (STT)**: Uses the Deepgram API for converting voice to text.
2. **Large Language Model (LLM)**: Uses OpenAI's GPT-3.5 Turbo.
3. **Text to Speech (TTS)**: Uses pyttsx3 library.

### File Structure

- `main.py`: Entry point for running the application.
- `voice_assistant.py`: Contains the core logic for managing audio input, processing, and output.
- `text_to_speech.py`: Contains the function for converting text to speech.


## Setup Instructions

### Prerequisites

Make sure you have the following tools and libraries installed:

- Python 3.7 or higher
- [Deepgram API Key](https://deepgram.com)
- [OpenAI API Key](https://platform.openai.com/) or replace it with any other llm
- Required Python packages: `pyaudio`, `websockets`, `pyttsx3`, `dotenv`, `openai`, `ffmpeg-python`

### Installation

1. **Clone the Repository**

   ```bash
   
   cd voice-assistant
   ```

2. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**

   Create a `.env` file in the root directory with the following content:

   ```bash
   DEEPGRAM_API_KEY=""
   OPENAI_API_KEY=""
   ```

4. **Install FFmpeg**

   Make sure `ffmpeg` is installed and added to your system's PATH. [FFmpeg Download](https://ffmpeg.org/download.html).

### Usage
**How It Works**

   - The assistant starts listening to audio from your microphone.
   - The audio is streamed to Deepgram for real-time transcription.
   - Once transcription is complete, the text is sent to the OpenAI API (or chosen LLM).
   - The LLM response is converted to speech using the pyttsx3 library.
   - The audio response is played back to the user.


