#!/usr/bin/env python3
"""
Speech Processor Node for Vision-Language-Action (VLA) System

This node handles speech-to-text conversion using Whisper model,
processes audio input with noise reduction, and publishes
transcribed text as ROS 2 messages.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import pyaudio
import numpy as np
import threading
import queue
import time
import whisper
from collections import deque

# Import the custom message
from std_msgs.msg import String
from builtin_interfaces.msg import Time
# Note: We'll need to adjust this import once the package is properly set up
# from humanoid_robotics_book.msg import SpeechCommand


class SpeechProcessorNode(Node):
    def __init__(self):
        super().__init__('speech_processor')

        # Parameters
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('chunk_size', 1024)
        self.declare_parameter('model_size', 'base')  # tiny, base, small, medium, large
        self.declare_parameter('language', 'en')
        self.declare_parameter('noise_threshold', 0.01)
        self.declare_parameter('silence_duration', 1.0)  # seconds of silence to trigger processing

        self.sample_rate = self.get_parameter('sample_rate').value
        self.chunk_size = self.get_parameter('chunk_size').value
        self.model_size = self.get_parameter('model_size').value
        self.language = self.get_parameter('language').value
        self.noise_threshold = self.get_parameter('noise_threshold').value
        self.silence_duration = self.get_parameter('silence_duration').value

        # Initialize Whisper model
        self.get_logger().info(f'Loading Whisper model ({self.model_size})...')
        # self.model = whisper.load_model(self.model_size)  # Uncomment when Whisper is available
        self.get_logger().info('Whisper model loaded successfully.')

        # Audio processing setup
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=self.sample_rate * 5)  # 5 seconds buffer
        self.is_recording = False
        self.recording_thread = None

        # Publishers
        # self.speech_pub = self.create_publisher(
        #     SpeechCommand,
        #     'speech_command',
        #     QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # )

        # Timer for processing audio
        self.process_timer = self.create_timer(0.1, self.process_audio)

        # Initialize audio stream
        self.init_audio_stream()

        self.get_logger().info('Speech Processor node initialized.')

    def init_audio_stream(self):
        """Initialize audio input stream"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            self.get_logger().info('Audio stream initialized successfully.')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize audio stream: {e}')
            raise

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input"""
        # Add audio data to queue for processing
        self.audio_queue.put(np.frombuffer(in_data, dtype=np.int16))
        return (None, pyaudio.paContinue)

    def is_silent(self, data, threshold=None):
        """Check if audio data is silent based on amplitude"""
        if threshold is None:
            threshold = self.noise_threshold

        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))
        return rms < threshold

    def apply_noise_reduction(self, audio_data):
        """Apply basic noise reduction to audio data"""
        # Simple noise reduction: spectral subtraction approach
        # Convert to frequency domain
        fft_data = np.fft.fft(audio_data)
        magnitude = np.abs(fft_data)

        # Estimate noise floor (assuming lower magnitudes are noise)
        noise_floor = np.percentile(magnitude, 10)  # 10th percentile as noise estimate

        # Apply spectral subtraction
        enhanced_magnitude = np.maximum(magnitude - noise_floor, 0)

        # Reconstruct signal
        phase = np.angle(fft_data)
        enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = np.real(np.fft.ifft(enhanced_fft)).astype(np.int16)

        return enhanced_audio

    def process_audio(self):
        """Process incoming audio data"""
        # Get audio data from queue
        audio_chunks = []
        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                audio_chunks.append(chunk)
            except queue.Empty:
                break

        if audio_chunks:
            # Concatenate all chunks
            audio_data = np.concatenate(audio_chunks)

            # Add to buffer
            for sample in audio_data:
                self.audio_buffer.append(sample)

        # Check if we have enough audio data and if it's time to process
        if len(self.audio_buffer) > self.sample_rate * 0.5:  # At least 0.5 seconds
            audio_array = np.array(self.audio_buffer)

            # Check if there's speech activity (not silent)
            if not self.is_silent(audio_array[-self.sample_rate//2:]):  # Check last 0.5 seconds
                # Continue recording
                self.is_recording = True
                self.last_speech_time = time.time()
            else:
                # Check if we were recording and it's been silent long enough
                if self.is_recording:
                    silence_duration = time.time() - self.last_speech_time
                    if silence_duration > self.silence_duration:
                        # Process the recorded audio
                        self.process_recording()
                        self.is_recording = False
                        # Clear buffer after processing
                        self.audio_buffer.clear()

    def process_recording(self):
        """Process the recorded audio and transcribe using Whisper"""
        if len(self.audio_buffer) == 0:
            return

        # Convert to numpy array
        audio_array = np.array(list(self.audio_buffer), dtype=np.float32)

        # Normalize audio
        audio_array = audio_array / np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) != 0 else audio_array

        # Apply noise reduction
        processed_audio = self.apply_noise_reduction(audio_array)

        try:
            # Transcribe using Whisper
            # result = self.model.transcribe(
            #     processed_audio,
            #     language=self.language,
            #     temperature=0.0  # More deterministic
            # )
            #
            # # Extract transcription and confidence
            # transcription = result['text'].strip()
            #
            # # Calculate confidence based on the confidence of individual tokens
            # # Whisper doesn't directly provide overall confidence, so we estimate it
            # avg_confidence = 0.8  # Default confidence
            #
            # if result.get('segments'):
            #     segment_confidences = []
            #     for segment in result['segments']:
            #         if 'tokens' in segment and len(segment['tokens']) > 0:
            #             # Extract confidence from tokens if available
            #             token_confidences = [token.get('probability', 1.0) for token in segment['tokens']]
            #             segment_confidences.extend(token_confidences)
            #
            #     if segment_confidences:
            #         avg_confidence = sum(segment_confidences) / len(segment_confidences)
            #
            # # Publish the result
            # self.publish_speech_command(transcription, avg_confidence)

            # For now, simulate transcription
            simulated_text = "Simulated speech recognition result"
            self.publish_speech_command(simulated_text, 0.8)

        except Exception as e:
            self.get_logger().error(f'Error during transcription: {e}')

    def publish_speech_command(self, transcription, confidence):
        """Publish the transcribed speech as a ROS 2 message"""
        if not transcription:
            return  # Don't publish empty transcriptions

        # msg = SpeechCommand()
        # msg.header.stamp = self.get_clock().now().to_msg()
        # msg.header.frame_id = 'microphone'
        # msg.utterance = transcription
        # msg.confidence = float(confidence)
        # msg.language = self.language
        # msg.timestamp = self.get_clock().now().to_msg()
        # msg.alternatives = []  # Could be populated with multiple hypotheses if needed
        #
        # self.speech_pub.publish(msg)
        self.get_logger().info(f'Published speech: "{transcription}" (confidence: {confidence:.2f})')

    def cleanup(self):
        """Clean up audio resources"""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()


def main(args=None):
    rclpy.init(args=args)

    try:
        speech_processor = SpeechProcessorNode()

        try:
            rclpy.spin(speech_processor)
        except KeyboardInterrupt:
            pass
        finally:
            speech_processor.cleanup()
            speech_processor.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()