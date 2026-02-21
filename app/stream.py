"""
Audio streaming module for microphone testing.

This module provides a callback-based audio stream using sounddevice
to test microphone input, particularly for AirPods on macOS.
"""

import sounddevice as sd
import numpy as np
import time
import sys
from typing import Optional


class MicrophoneStream:
    """Callback-based microphone stream with RMS volume monitoring."""
    
    def __init__(self, device_id: Optional[int] = None, samplerate: int = 16000):
        """
        Initialize microphone stream.
        
        Args:
            device_id: Input device ID (None for default)
            samplerate: Sample rate in Hz (default: 16000)
        """
        self.device_id = device_id
        self.samplerate = samplerate
        self.channels = 1  # Mono
        self.rms_values = []
        self.last_print_time = 0
        self.print_interval = 0.5  # Print every 0.5 seconds
        self.stream = None
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function called for each audio block."""
        if status:
            print(f"Status: {status}", file=sys.stderr)
        
        # Calculate RMS (Root Mean Square) volume
        rms = np.sqrt(np.mean(indata**2))
        self.rms_values.append(rms)
        
        # Print volume meter every 0.5 seconds
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_interval:
            # Convert RMS to dB (avoid log(0))
            rms_db = 20 * np.log10(max(rms, 1e-10))
            
            # Create a simple ASCII volume bar (0-50 dB range)
            bar_length = 40
            normalized = max(0, min(1, (rms_db + 60) / 50))  # Normalize -60 to -10 dB
            filled = int(bar_length * normalized)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\rRMS: {rms_db:6.2f} dB  [{bar}]", end='', flush=True)
            self.last_print_time = current_time
    
    def start(self, duration: float = 15.0):
        """
        Start streaming audio for specified duration.
        
        Args:
            duration: Duration in seconds (default: 15.0)
        """
        print(f"\n🎤 Starting microphone stream...")
        print(f"   Device ID: {self.device_id if self.device_id is not None else 'Default'}")
        print(f"   Sample rate: {self.samplerate} Hz")
        print(f"   Channels: {self.channels} (Mono)")
        print(f"   Duration: {duration} seconds")
        print(f"\n📊 Live RMS Volume Meter (updates every 0.5s):\n")
        
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
                time.sleep(duration)
            
            # Final statistics
            avg_rms = np.mean(self.rms_values) if self.rms_values else 0
            max_rms = np.max(self.rms_values) if self.rms_values else 0
            avg_db = 20 * np.log10(max(avg_rms, 1e-10))
            max_db = 20 * np.log10(max(max_rms, 1e-10))
            
            print(f"\n\n✅ Stream completed successfully!")
            print(f"   Average RMS: {avg_db:.2f} dB")
            print(f"   Peak RMS: {max_db:.2f} dB")
            print(f"   Total samples: {len(self.rms_values)}\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Stream interrupted by user\n")
        except Exception as e:
            print(f"\n\n❌ Error during streaming: {e}\n")
            raise


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
    print("=" * 80)
    print("🎤 vox-core Microphone Stream Test")
    print("=" * 80)
    
    # List devices and get selection
    device_id = get_device_selection()
    
    # device_id can be None (default) or an integer (selected device)
    # None is valid for default device, so we proceed
    
    # Create and start stream
    stream = MicrophoneStream(device_id=device_id, samplerate=16000)
    
    try:
        stream.start(duration=15.0)
    except Exception as e:
        print(f"\n❌ Failed to start stream: {e}\n")
        print_troubleshooting_tips()
        sys.exit(1)


def print_troubleshooting_tips():
    """Print troubleshooting tips for common issues."""
    print("\n" + "=" * 80)
    print("🔧 TROUBLESHOOTING TIPS")
    print("=" * 80)
    print("""
1. No input devices found:
   - Check System Preferences > Sound > Input
   - Ensure microphone is connected and recognized by macOS
   - Try disconnecting and reconnecting AirPods

2. Permission denied errors:
   - macOS requires microphone permission for Terminal/Python
   - Go to: System Preferences > Security & Privacy > Privacy > Microphone
   - Enable Terminal (or your Python IDE) to access microphone

3. Device not working:
   - Verify device is selected in System Preferences > Sound > Input
   - Try selecting a different device ID from the list
   - Check if device supports 16000 Hz sample rate

4. Low or no volume detected:
   - Speak closer to the microphone
   - Check microphone volume in System Preferences > Sound > Input
   - Ensure AirPods are connected and set as input device
   - Try increasing input volume slider

5. Audio callback errors:
   - Ensure no other application is using the microphone
   - Try closing other audio applications
   - Restart the stream

6. Sample rate issues:
   - Some devices may not support exactly 16000 Hz
   - sounddevice will resample automatically if needed
   - Check device capabilities: python3 -c "import sounddevice as sd; print(sd.query_devices())"

For more help, check sounddevice documentation:
https://python-sounddevice.readthedocs.io/
""")


if __name__ == "__main__":
    main()
