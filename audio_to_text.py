import whisper
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[ logging.StreamHandler() ]
)
logger = logging.getLogger(__name__)

def transcribe_audio_whisper(input_audio, output_file="transcript.txt"):
    model = whisper.load_model("turbo")
    result = model.transcribe(input_audio)
    logger.info("Transcription completed successfully.")
    
    text = result["text"]
    
    # Write transcript to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    logger.info(f"Transcript saved to: {output_file}")
    return text, output_file

if __name__ == '__main__':
    text, transcript_file = transcribe_audio_whisper("audio.wav")
    print(f"Transcript saved to: {transcript_file}")
    print(f"Text preview: {text[:200]}...")
