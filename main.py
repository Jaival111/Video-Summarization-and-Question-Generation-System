from extract_audio import download_youtube_video, extract_audio
from audio_to_text import transcribe_audio_whisper
from summarize import lexrank_summary, bart_summary

video_link = "https://youtu.be/MwZwr5Tvyxo?si=Z-_QfI4ZorD9L5ju"

if __name__ == '__main__':

    download_youtube_video(video_link)

    extract_audio("video.mp4")
    
    text = transcribe_audio_whisper("audio.wav")

    with open("transcription.txt", "r", encoding="utf-8") as f:
        text = f.read()

    abstractive_summary = bart_summary(text)

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(abstractive_summary)