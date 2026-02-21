"""
Live transcription module using Whisper with rolling buffer.

This module provides callback-based audio streaming with periodic
Whisper transcription for real-time speech-to-text.
"""

import sounddevice as sd
import numpy as np
import whisper
import time
import sys
import threading
import argparse
from typing import Optional, List


class LiveTranscriber:
    """Callback-based audio stream with rolling buffer transcription."""
    
    def __init__(
        self,
        device_id: Optional[int] = None,
        samplerate: int = 16000,
        chunk_seconds: float = 6.0,
        model_name: str = "base"
    ):
        """
        Initialize live transcriber.
        
        Args:
            device_id: Input device ID (None for default)
            samplerate: Sample rate in Hz (default: 16000)
            chunk_seconds: Seconds of audio to accumulate before transcribing
            model_name: Whisper model name (tiny, base, small, medium, large)
        """
        self.device_id = device_id
        self.samplerate = samplerate
        self.channels = 1  # Mono
        self.chunk_seconds = chunk_seconds
        self.model_name = model_name
        
        # Audio buffer (accumulates chunk_seconds of audio)
        self.buffer: List[float] = []
        self.buffer_lock = threading.Lock()
        self.buffer_max_samples = int(samplerate * chunk_seconds)
        
        # Transcription state
        self.model = None
        self.transcription_thread = None
        self.running = False
        self.stream = None
        self.start_time = None
        
        # Statistics
        self.transcription_count = 0
        
    def load_model(self):
        """Load Whisper model (CPU mode, fp16=False)."""
        print(f"📦 Loading Whisper model: {self.model_name} (CPU mode)...")
        try:
            self.model = whisper.load_model(self.model_name, device="cpu")
            print(f"✅ Model loaded successfully\n")
        except Exception as e:
            print(f"❌ Failed to load model: {e}\n")
            raise
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function called for each audio block."""
        if status:
            print(f"Status: {status}", file=sys.stderr)
        
        # Append audio data to buffer (convert to 1D if needed)
        audio_data = indata[:, 0] if indata.ndim > 1 else indata
        
        with self.buffer_lock:
            self.buffer.extend(audio_data)
            # Keep buffer size limited (safety check)
            if len(self.buffer) > self.buffer_max_samples * 2:
                # If buffer grows too large, keep only the most recent chunk
                self.buffer = self.buffer[-self.buffer_max_samples:]
    
    def transcribe_buffer(self):
        """Transcribe the current audio buffer and clear it."""
        with self.buffer_lock:
            if len(self.buffer) < self.buffer_max_samples:
                # Not enough audio accumulated yet
                return None
            
            # Get exactly chunk_seconds worth of audio
            audio_array = np.array(self.buffer[:self.buffer_max_samples], dtype=np.float32)
            
            # Clear the transcribed portion, keep any overflow for next chunk
            self.buffer = self.buffer[self.buffer_max_samples:]
        
        # Transcribe using Whisper
        try:
            result = self.model.transcribe(
                audio_array,
                fp16=False,  # CPU mode
                language="en",  # Optional: can be None for auto-detect
                verbose=False
            )
            return result.get("text", "").strip()
        except Exception as e:
            print(f"⚠️  Transcription error: {e}", file=sys.stderr)
            return None
    
    def transcription_worker(self):
        """Worker thread that periodically transcribes the buffer."""
        while self.running:
            time.sleep(self.chunk_seconds)
            
            if not self.running:
                break
            
            # Get transcript
            transcript = self.transcribe_buffer()
            
            if transcript and transcript:
                self.transcription_count += 1
                elapsed = time.time() - self.start_time
                
                # Print transcript with timestamp
                timestamp = f"[{elapsed:7.1f}s]"
                print(f"{timestamp} {transcript}")
                sys.stdout.flush()
    
    def start(self):
        """Start streaming and transcription."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"🎤 Starting live transcription...")
        print(f"   Device ID: {self.device_id if self.device_id is not None else 'Default'}")
        print(f"   Sample rate: {self.samplerate} Hz")
        print(f"   Channels: {self.channels} (Mono)")
        print(f"   Chunk duration: {self.chunk_seconds} seconds")
        print(f"   Model: {self.model_name} (CPU)")
        print(f"\n📝 Transcripts (updates every {self.chunk_seconds}s):\n")
        print("-" * 80)
        
        self.running = True
        self.start_time = time.time()
        
        # Start transcription worker thread
        self.transcription_thread = threading.Thread(
            target=self.transcription_worker,
            daemon=True
        )
        self.transcription_thread.start()
        
        try:
            # Create and start the input stream
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=self.samplerate,
                callback=self.audio_callback,
                dtype=np.float32
            )
            
            with self.stream:
                # Keep streaming until interrupted
                while self.running:
                    time.sleep(0.1)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Error during streaming: {e}")
            raise
        finally:
            self.stop()
    
    def stop(self):
        """Stop streaming and transcription."""
        self.running = False
        
        if self.stream:
            self.stream.stop()
        
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=2.0)
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"\n✅ Transcription stopped")
        print(f"   Total time: {elapsed:.1f} seconds")
        print(f"   Transcriptions: {self.transcription_count}\n")


def list_input_devices():
    """List all available input devices."""
    print("\n📋 Available Input Devices:\n")
    print(f"{'ID':<5} {'Name':<40} {'Channels':<10} {'Sample Rate':<15}")
    print("-" * 80)
    
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append(i)
            channels = device['max_input_channels']
            samplerate = device['default_samplerate']
            name = device['name']
            print(f"{i:<5} {name:<40} {channels:<10} {samplerate:<15.0f}")
    
    print(f"\nTotal input devices found: {len(input_devices)}\n")
    return input_devices


def get_device_selection() -> Optional[int]:
    """Prompt user to select a device by ID."""
    input_devices = list_input_devices()
    
    if not input_devices:
        print("❌ No input devices found!")
        return None
    
    print("💡 Tip: Use device ID from the list above, or press Enter for default device\n")
    
    while True:
        try:
            user_input = input("Enter device ID (or press Enter for default): ").strip()
            
            if not user_input:
                print("Using default input device\n")
                return None
            
            device_id = int(user_input)
            
            if device_id in input_devices:
                device_info = sd.query_devices(device_id)
                print(f"\n✅ Selected: {device_info['name']}\n")
                return device_id
            else:
                print(f"❌ Invalid device ID. Please choose from: {input_devices}\n")
                
        except ValueError:
            print("❌ Please enter a valid number or press Enter for default\n")
        except KeyboardInterrupt:
            print("\n\nCancelled by user\n")
            return None


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Live transcription using Whisper with rolling buffer"
    )
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=6.0,
        help="Seconds of audio to accumulate before transcribing (default: 6.0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (default: base)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device ID (default: system default)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input devices and exit"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎙️  vox-core Live Transcription")
    print("=" * 80)
    
    if args.list_devices:
        list_input_devices()
        return
    
    # Get device selection if not provided
    device_id = args.device
    if device_id is None:
        device_id = get_device_selection()
        if device_id is None and device_id != 0:
            # User cancelled or chose default
            pass
    
    # Create transcriber
    transcriber = LiveTranscriber(
        device_id=device_id,
        samplerate=16000,
        chunk_seconds=args.chunk_seconds,
        model_name=args.model
    )
    
    try:
        # Load model
        transcriber.load_model()
        
        # Start transcription
        transcriber.start()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user\n")
        transcriber.stop()
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print_troubleshooting_tips()
        sys.exit(1)


def print_troubleshooting_tips():
    """Print troubleshooting tips for common issues."""
    print("\n" + "=" * 80)
    print("🔧 TROUBLESHOOTING TIPS")
    print("=" * 80)
    print("""
1. Model download issues:
   - Whisper models are downloaded automatically on first use
   - Ensure you have internet connection for first run
   - Models are cached in ~/.cache/whisper/

2. Slow transcription:
   - Use smaller model: --model tiny or --model base
   - Increase chunk_seconds to transcribe less frequently
   - Ensure you're using CPU mode (fp16=False)

3. No transcriptions appearing:
   - Check microphone is working and picking up audio
   - Verify device selection is correct
   - Ensure sufficient audio in buffer (speak clearly)

4. Permission errors:
   - macOS requires microphone permission
   - System Preferences > Security & Privacy > Privacy > Microphone
   - Enable Terminal/Python to access microphone

5. Audio buffer issues:
   - If transcriptions are delayed, reduce chunk_seconds
   - If transcriptions are too frequent, increase chunk_seconds
   - Default 6 seconds is a good balance

6. Memory issues with large models:
   - Use smaller models (tiny, base, small)
   - Large models require significant RAM
   - Base model is recommended for CPU usage

For more help, check Whisper documentation:
https://github.com/openai/whisper
""")


if __name__ == "__main__":
    main()
