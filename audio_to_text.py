import whisper
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[ logging.StreamHandler() ]
)
logger = logging.getLogger(__name__)

def transcribe_audio_whisper(input_audio):
    model = whisper.load_model("turbo")
    result = model.transcribe(input_audio)
    logger.info("Transcription completed successfully.")
    return result["text"]

def transcribe_audio_indic(input_audio):
    pass

if __name__ == '__main__':
    text = transcribe_audio_whisper("audio.wav")
    with open("transcription.txt", "w") as f:
        f.write(text)
