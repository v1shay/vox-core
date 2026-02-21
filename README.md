# vox-core

Real-time AirPods mic → Whisper transcription → LLM structured output system.

## Prerequisites

- Python 3.8+
- ffmpeg (required for Whisper on macOS)

### Install ffmpeg on macOS

```bash
brew install ffmpeg
```

## Setup

### STEP 1 — Create Project and Virtual Environment

```bash
# Create project folder (if not already created)
mkdir -p vox-core
cd vox-core

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### STEP 2 — Git Setup

```bash
# Initialize git repository
git init

# Add .gitignore (already configured for venv, __pycache__, .env, out/)
git add .gitignore

# Add all project files
git add .

# Create first commit
git commit -m "Initial commit: vox-core project structure"

# Connect to remote repository
git remote add origin https://github.com/v1shay/vox-core.git

# Push to main branch
git branch -M main
git push -u origin main
```

## Verification Checklist

### Verify Microphone Devices

```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

### Verify Whisper Installation

```bash
python3 -c "import whisper; print('Whisper version:', whisper.__version__)"
```

### Verify Git Remote

```bash
git remote -v
```

Expected output should show:
```
origin  https://github.com/v1shay/vox-core.git (fetch)
origin  https://github.com/v1shay/vox-core.git (push)
```

## Project Structure

```
vox-core/
│
├── app/
│   ├── __init__.py
│   ├── stream.py
│   ├── transcribe.py
│   ├── llm.py
│   └── run_live.py
│
├── out/
│
├── requirements.txt
├── README.md
└── .gitignore
```
