#!/usr/bin/env python3
"""
Audio Capture Module for Shield Vision
Handles real-time audio capture and processing for threat detection
"""

import threading
import queue
import time
import numpy as np
from collections import deque

# Try to import audio libraries
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️  PyAudio not available - audio capture disabled")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  Librosa not available - audio analysis limited")

class AudioCapture:
    """
    Real-time audio capture and processing class
    """
    
    def __init__(self, sample_rate=22050, chunk_size=1024, channels=1):
        """
        Initialize audio capture
        
        Args:
            sample_rate: Audio sample rate (Hz)
            chunk_size: Number of samples per chunk
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        
        # Audio stream and threading
        self.audio = None
        self.stream = None
        self.is_recording = False
        self.audio_thread = None
        
        # Audio data storage
        self.audio_queue = queue.Queue(maxsize=100)
        self.audio_buffer = deque(maxlen=100)  # Store recent audio chunks
        
        # Initialize PyAudio if available
        if PYAUDIO_AVAILABLE:
            try:
                self.audio = pyaudio.PyAudio()
                print("✅ PyAudio initialized successfully")
            except Exception as e:
                print(f"❌ Error initializing PyAudio: {e}")
                self.audio = None
        
    def start_recording(self):
        """Start audio recording in a separate thread"""
        if not PYAUDIO_AVAILABLE or self.audio is None:
            print("⚠️  Audio recording not available")
            return False
        
        if self.is_recording:
            print("⚠️  Audio recording already active")
            return True
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            
            print(f"🎤 Audio recording started - {self.sample_rate}Hz, {self.channels} channel(s)")
            return True
            
        except Exception as e:
            print(f"❌ Error starting audio recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop audio recording"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                print("🔴 Audio recording stopped")
            except Exception as e:
                print(f"⚠️  Error stopping audio stream: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert audio data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Add to queue for processing
        try:
            self.audio_queue.put_nowait(audio_data)
            self.audio_buffer.append(audio_data)
        except queue.Full:
            # Remove oldest item if queue is full
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_data)
            except queue.Empty:
                pass
        
        return (in_data, pyaudio.paContinue)
    
    def get_latest_audio(self, duration_seconds=1.0):
        """
        Get the latest audio data for analysis
        
        Args:
            duration_seconds: Duration of audio to return (seconds)
            
        Returns:
            numpy array of audio data or None if no data available
        """
        if not self.audio_buffer:
            return None
        
        # Calculate how many chunks we need for the requested duration
        chunks_needed = int((duration_seconds * self.sample_rate) / self.chunk_size)
        chunks_needed = min(chunks_needed, len(self.audio_buffer))
        
        if chunks_needed == 0:
            return None
        
        # Get the most recent chunks
        recent_chunks = list(self.audio_buffer)[-chunks_needed:]
        
        # Concatenate chunks into a single array
        audio_data = np.concatenate(recent_chunks)
        
        # Normalize to [-1, 1] range
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        return audio_data
    
    def get_audio_features(self, audio_data):
        """
        Extract basic audio features for analysis
        
        Args:
            audio_data: numpy array of audio samples
            
        Returns:
            dict of audio features
        """
        if audio_data is None or len(audio_data) == 0:
            return {
                'volume': 0.0,
                'energy': 0.0,
                'zero_crossing_rate': 0.0,
                'spectral_centroid': 0.0
            }
        
        # Basic features
        volume = np.sqrt(np.mean(audio_data ** 2))  # RMS volume
        energy = np.sum(audio_data ** 2)
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.sign(audio_data)))[0]
        zcr = len(zero_crossings) / len(audio_data)
        
        # Spectral centroid (if librosa is available)
        spectral_centroid = 0.0
        if LIBROSA_AVAILABLE:
            try:
                centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
                spectral_centroid = np.mean(centroid)
            except Exception as e:
                print(f"Error calculating spectral centroid: {e}")
        
        return {
            'volume': float(volume),
            'energy': float(energy),
            'zero_crossing_rate': float(zcr),
            'spectral_centroid': float(spectral_centroid)
        }
    
    def is_audio_active(self, threshold=0.005):  # Much lower threshold
        """
        Check if there's active audio (above threshold)

        Args:
            threshold: Volume threshold for considering audio as active

        Returns:
            bool: True if audio is active
        """
        audio_data = self.get_latest_audio(0.5)  # Check last 0.5 seconds
        if audio_data is None:
            return False

        volume = np.sqrt(np.mean(audio_data ** 2))
        return volume > threshold
    
    def cleanup(self):
        """Clean up audio resources"""
        self.stop_recording()
        
        if self.audio:
            try:
                self.audio.terminate()
                print("🔧 Audio resources cleaned up")
            except Exception as e:
                print(f"⚠️  Error cleaning up audio: {e}")


class AudioAnalyzer:
    """
    Audio analysis class for threat detection
    """
    
    def __init__(self, audio_capture):
        """
        Initialize audio analyzer

        Args:
            audio_capture: AudioCapture instance
        """
        self.audio_capture = audio_capture
        self.threat_threshold = 0.3  # Much lower threshold for more sensitivity
        self.analysis_history = deque(maxlen=10)
    
    def analyze_for_threats(self):
        """
        Analyze current audio for potential threats
        
        Returns:
            dict: Analysis results with threat detection
        """
        # Get recent audio data
        audio_data = self.audio_capture.get_latest_audio(1.0)
        
        if audio_data is None:
            return {
                'threat_detected': False,
                'confidence': 0.0,
                'features': {},
                'status': 'no_audio'
            }
        
        # Extract features
        features = self.audio_capture.get_audio_features(audio_data)
        
        # Simple threat detection based on audio characteristics
        # This is a simplified heuristic - in production, use the trained model
        threat_score = 0.0

        # Base score from volume (more sensitive)
        volume_score = min(features['volume'] * 5.0, 1.0)  # Amplify volume sensitivity
        threat_score += volume_score * 0.4

        # High volume might indicate shouting/distress (lowered threshold)
        if features['volume'] > 0.05:  # Much lower threshold
            threat_score += 0.3

        # Very high volume indicates likely threat
        if features['volume'] > 0.15:
            threat_score += 0.4

        # High energy might indicate aggressive sounds (lowered threshold)
        if features['energy'] > 0.01:  # Much lower threshold
            threat_score += 0.2

        # High zero crossing rate might indicate distress calls (lowered threshold)
        if features['zero_crossing_rate'] > 0.05:  # Lower threshold
            threat_score += 0.2

        # Sudden volume changes might indicate conflict
        if len(self.analysis_history) > 0:
            prev_volume = self.analysis_history[-1]['features'].get('volume', 0)
            volume_change = abs(features['volume'] - prev_volume)
            if volume_change > 0.05:  # Much lower threshold
                threat_score += 0.3

        # Normalize threat score
        threat_score = min(threat_score, 1.0)
        
        # Determine if threat is detected
        threat_detected = threat_score > self.threat_threshold
        
        result = {
            'threat_detected': threat_detected,
            'confidence': threat_score,
            'features': features,
            'status': 'active' if self.audio_capture.is_audio_active() else 'quiet'
        }
        
        # Store in history
        self.analysis_history.append(result)
        
        return result


# Global audio capture instance
audio_capture = None
audio_analyzer = None

def initialize_audio():
    """Initialize global audio capture and analyzer"""
    global audio_capture, audio_analyzer
    
    if not PYAUDIO_AVAILABLE:
        print("⚠️  Audio capture not available - PyAudio not installed")
        return False
    
    try:
        audio_capture = AudioCapture()
        audio_analyzer = AudioAnalyzer(audio_capture)
        
        # Start recording
        if audio_capture.start_recording():
            print("✅ Audio capture and analysis initialized")
            return True
        else:
            print("❌ Failed to start audio recording")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing audio: {e}")
        return False

def get_audio_analysis():
    """Get current audio analysis results"""
    global audio_analyzer
    
    if audio_analyzer is None:
        return {
            'threat_detected': False,
            'confidence': 0.0,
            'features': {},
            'status': 'not_available'
        }
    
    return audio_analyzer.analyze_for_threats()

def cleanup_audio():
    """Clean up audio resources"""
    global audio_capture
    
    if audio_capture:
        audio_capture.cleanup()

# Test function
def test_audio():
    """Test audio capture functionality"""
    print("Testing audio capture...")
    
    if not PYAUDIO_AVAILABLE:
        print("❌ PyAudio not available")
        return False
    
    # Initialize audio
    if initialize_audio():
        print("✅ Audio initialized successfully")
        
        # Test for a few seconds
        print("🎤 Recording for 5 seconds...")
        time.sleep(5)
        
        # Get analysis
        analysis = get_audio_analysis()
        print(f"📊 Audio analysis: {analysis}")
        
        # Cleanup
        cleanup_audio()
        return True
    else:
        print("❌ Audio initialization failed")
        return False

if __name__ == "__main__":
    test_audio()
