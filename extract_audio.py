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
        'format': 'bv*+ba/best',
        'outtmpl': output_path,
        'quiet': True,
        'merge_output_format': 'mp4',
        'force_ipv4': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.youtube.com/'
        }
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    logger.info("Video downloaded successfully.")
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
        logger.info("Audio extracted successfully.")
        return output_audio
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e}")
        raise


if __name__ == '__main__':
    video_path = download_youtube_video("https://youtu.be/VO6XEQIsCoM?si=_olid6rVQRSlds5g")
    audio_path = extract_audio(video_path)
    print(f"Video saved to: {video_path}")
    print(f"Audio saved to: {audio_path}")