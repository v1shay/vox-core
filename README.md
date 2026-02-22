# vox-agent
> A real-time hardware-integrated audio intelligence system built in Python, transforming live speech from devices like AirPods or Mac microphones into structured output using CoreAudio, Whisper, and OpenAI API integration.

---

## Features
- Real-time microphone streaming via macOS CoreAudio (AirPods/Mac input)  
- Incremental speech-to-text using Whisper (local CPU inference)  
- Mode-based structured formatting (meeting, study, recitation, interview)  
- Secure API key management via environment variables  
- Markdown session export for persistent notes  
- CLI-based configuration for chunk size and formatting interval  

---

## Why This Exists
Voice notes and transcripts are often unstructured and difficult to review. Context is lost in raw text, and important decisions or insights are buried in paragraphs of speech.

This project exists to explore how live audio input can be transformed into structured intelligence in real time. Instead of storing raw recordings or flat transcripts, the system converts speech into organized summaries, action items, and conceptual breakdowns as it is spoken.

The goal is not complexity, but clarity through structured reasoning.

---

## How It Works
The system follows a layered, modular architecture.

1. Microphone input is captured using CoreAudio via the `sounddevice` library  
2. Audio frames are buffered and processed in rolling chunks  
3. Whisper performs local speech recognition on each chunk  
4. The transcript is accumulated incrementally  
5. At fixed intervals, the transcript is sent to the OpenAI API for structured semantic formatting  
6. Structured output is printed to the console and saved as Markdown  

The system runs entirely from the command line and requires no frontend or backend services.

---

## Tech Stack
- **Language:** Python  
- **Audio Streaming:** macOS CoreAudio (sounddevice)  
- **Speech Recognition:** OpenAI Whisper (local inference)  
- **LLM Integration:** OpenAI API (gpt-4o-mini)  
- **Environment Management:** python-dotenv  
- **Version Control:** Git  

---

## Project Structure
```text
vox-agent/
├── app/
│   ├── stream.py
│   ├── transcribe.py
│   ├── llm.py
│   └── run_live.py
├── out/
├── requirements.txt
├── .env
└── README.md

