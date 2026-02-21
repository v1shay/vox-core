"""
Live transcription and structured output runner.

Combines audio streaming, Whisper transcription, and OpenAI LLM
structured formatting for real-time note-taking.
"""

import sys
import time
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.transcribe import LiveTranscriber, list_input_devices, get_device_selection
from app.llm import format_transcript, OPENAI_API_KEY


class LiveRunner:
    """Combined live transcription and LLM structuring."""
    
    def __init__(
        self,
        mode: str,
        device_id: Optional[int] = None,
        chunk_seconds: float = 6.0,
        summarize_every: float = 25.0,
        whisper_model: str = "base"
    ):
        """
        Initialize live runner.
        
        Args:
            mode: Output mode (meeting, study, recitation, interview)
            device_id: Input device ID (None for default)
            chunk_seconds: Seconds of audio per transcription chunk
            summarize_every: Seconds between LLM formatting calls
            whisper_model: Whisper model name
        """
        self.mode = mode
        self.summarize_every = summarize_every
        
        # Initialize transcriber
        self.transcriber = LiveTranscriber(
            device_id=device_id,
            samplerate=16000,
            chunk_seconds=chunk_seconds,
            model_name=whisper_model
        )
        
        # Transcript accumulation
        self.accumulated_transcript = []
        self.last_summarize_time = None
        self.start_time = None
        
        # Output file
        self.output_dir = Path("out")
        self.output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"session_notes_{timestamp}.md"
        
        # Initialize output file with header
        self._init_output_file()
        
        # Statistics
        self.transcript_count = 0
        self.llm_call_count = 0
        
    def on_transcript(self, transcript: str, elapsed: float):
        """
        Callback for new transcript chunks.
        
        Args:
            transcript: New transcript text
            elapsed: Elapsed time in seconds
        """
        if not transcript or len(transcript.strip()) < 10:
            return
        
        self.transcript_count += 1
        self.accumulated_transcript.append(transcript)
        
        # Print raw transcript
        timestamp = f"[{elapsed:7.1f}s]"
        print(f"{timestamp} {transcript}")
        sys.stdout.flush()
        
        # Check if we should summarize
        if self.last_summarize_time is None:
            self.last_summarize_time = time.time()
        
        time_since_last = time.time() - self.last_summarize_time
        
        if time_since_last >= self.summarize_every:
            self._summarize_and_save()
            self.last_summarize_time = time.time()
    
    def _summarize_and_save(self):
        """Summarize accumulated transcript and save to file."""
        if not self.accumulated_transcript:
            return
        
        # Combine accumulated transcripts
        full_transcript = " ".join(self.accumulated_transcript)
        
        if len(full_transcript.strip()) < 50:
            return  # Skip if too short
        
        print("\n" + "=" * 80)
        print("===== STRUCTURED UPDATE =====")
        print("=" * 80 + "\n")
        
        try:
            # Get structured output from LLM
            structured_output = format_transcript(self.mode, full_transcript)
            
            if structured_output:
                self.llm_call_count += 1
                
                # Print structured output
                print(structured_output)
                print("\n" + "=" * 80 + "\n")
                
                # Append to output file
                self._append_to_file(structured_output, full_transcript)
                
                # Clear accumulated transcript (keep only last chunk for continuity)
                if len(self.accumulated_transcript) > 1:
                    self.accumulated_transcript = [self.accumulated_transcript[-1]]
                else:
                    self.accumulated_transcript = []
            else:
                print("⚠️  Transcript too short, skipping LLM call\n")
                
        except ValueError as e:
            print(f"❌ Error: {e}\n")
            # Don't clear transcript on error, allow retry
        except Exception as e:
            print(f"❌ Unexpected error: {e}\n")
    
    def _append_to_file(self, structured_output: str, raw_transcript: str):
        """Append structured output to session notes file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- Update at {timestamp} (Elapsed: {elapsed:.1f}s) ---\n\n")
            f.write(structured_output)
            f.write(f"\n\n<!-- Raw transcript: {raw_transcript[:200]}... -->\n")
    
    def _init_output_file(self):
        """Initialize output file with header."""
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(f"# Session Notes - {self.mode.capitalize()} Mode\n\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {self.mode}\n\n")
            f.write("---\n\n")
    
    def _save_final_summary(self):
        """Save final summary on exit."""
        if not self.accumulated_transcript:
            return
        
        # Final summarize of any remaining transcript
        full_transcript = " ".join(self.accumulated_transcript)
        
        if len(full_transcript.strip()) >= 50:
            print("\n" + "=" * 80)
            print("===== FINAL STRUCTURED UPDATE =====")
            print("=" * 80 + "\n")
            
            try:
                structured_output = format_transcript(self.mode, full_transcript)
                
                if structured_output:
                    print(structured_output)
                    print("\n" + "=" * 80 + "\n")
                    self._append_to_file(structured_output, full_transcript)
            except Exception as e:
                print(f"⚠️  Final summary error: {e}\n")
    
    def start(self):
        """Start live transcription and structuring."""
        # Check API key
        if not OPENAI_API_KEY:
            print("❌ ERROR: OPENAI_API_KEY environment variable not set")
            print("\nPlease set it with:")
            print("  export OPENAI_API_KEY='your-key-here'")
            print("\nOr add it to your shell profile (~/.zshrc or ~/.bashrc)\n")
            sys.exit(1)
        
        # Load Whisper model
        self.transcriber.load_model()
        
        # Override transcriber's transcription worker to use our callback
        # This must be done before calling start() which creates the thread
        def custom_worker():
            """Custom worker that calls our on_transcript callback."""
            while self.transcriber.running:
                time.sleep(self.transcriber.chunk_seconds)
                
                if not self.transcriber.running:
                    break
                
                transcript = self.transcriber.transcribe_buffer()
                
                if transcript and transcript:
                    elapsed = time.time() - self.transcriber.start_time
                    self.on_transcript(transcript, elapsed)
        
        self.transcriber.transcription_worker = custom_worker
        
        # Print startup info
        print("=" * 80)
        print("🎙️  vox-core Live Transcription & Structuring")
        print("=" * 80)
        print(f"Mode: {self.mode}")
        print(f"Transcription chunk: {self.transcriber.chunk_seconds}s")
        print(f"LLM formatting: every {self.summarize_every}s")
        print(f"Output file: {self.output_file}")
        print("=" * 80 + "\n")
        
        self.start_time = time.time()
        
        try:
            # Start transcriber (this will start the stream and worker thread)
            self.transcriber.start()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        finally:
            self._save_final_summary()
            self.transcriber.stop()
            
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"\n✅ Session completed")
            print(f"   Total time: {elapsed:.1f} seconds")
            print(f"   Transcripts: {self.transcript_count}")
            print(f"   LLM calls: {self.llm_call_count}")
            print(f"   Notes saved to: {self.output_file}\n")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Live transcription with structured LLM output"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["meeting", "study", "recitation", "interview"],
        help="Output mode: meeting, study, recitation, or interview"
    )
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=6.0,
        help="Seconds of audio per transcription chunk (default: 6.0)"
    )
    parser.add_argument(
        "--summarize_every",
        type=float,
        default=25.0,
        help="Seconds between LLM formatting calls (default: 25.0)"
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
    
    # Create and start runner
    runner = LiveRunner(
        mode=args.mode,
        device_id=device_id,
        chunk_seconds=args.chunk_seconds,
        summarize_every=args.summarize_every,
        whisper_model=args.model
    )
    
    try:
        runner.start()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
