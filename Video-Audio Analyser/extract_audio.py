import subprocess
import os
import yt_dlp

def download_youtube_video(url, output_path="video.mp4"):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': output_path,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Video downloaded")
    return output_path

def extract_audio(video_path, output_audio="audio.wav"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        output_audio
    ]

    try:
        subprocess.run(command, check=True)
        print("Audio extracted successfully.")
    except subprocess.CalledProcessError as e:
        print("Error extracting audio:", e)

download_youtube_video("https://youtu.be/wHHxkWcqokY?si=t_cCSlqwSr-1v-KZ")
extract_audio("video.mp4")