# SpeechCommand.msg
# Message for speech-to-text processing in VLA system

std_msgs/Header header
string utterance          # The transcribed text from speech
float32 confidence        # Confidence score for the transcription (0.0 to 1.0)
string language           # Detected language code (e.g., 'en', 'es', 'fr')
builtin_interfaces/Time timestamp  # When the speech was processed
string[] alternatives     # Alternative transcriptions if available