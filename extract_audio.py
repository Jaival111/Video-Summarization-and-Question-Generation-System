import subprocess
import os
import logging
import yt_dlp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[ logging.StreamHandler() ]
)
logger = logging.getLogger(__name__)

def download_youtube_video(url, output_path="video.mp4"):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': output_path,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    logger.info("Video downloaded successfully.")

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
        logger.info("Audio extracted successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e}")


if __name__ == '__main__':
    download_youtube_video("https://youtu.be/MwZwr5Tvyxo?si=Z-_QfI4ZorD9L5ju")
    extract_audio("video.mp4")