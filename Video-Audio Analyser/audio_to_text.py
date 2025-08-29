import whisper

def transcribe_audio(input_audio):
    model = whisper.load_model("turbo")
    result = model.transcribe(input_audio)
    return result["text"]

text = transcribe_audio("audio.wav")

with open("transcription.txt", "w") as f:
    f.write(text)
