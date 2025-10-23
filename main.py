import streamlit as st
import os
from extract_audio import download_youtube_video, extract_audio
from audio_to_text import transcribe_audio_whisper
from summarize import bart_summary
from preprocess import save_chunks_to_json
from generate_quiz import generate_qna_with_references, generate_topic

st.set_page_config(page_title="🎬 YouTube / Video Summarizer", layout="centered")
st.title("🎥 YouTube / Video Summarizer")
st.write("Upload a video or paste a YouTube link to generate transcript and summary.")

# ========================
# 🔹 Cached Functions
# ========================
@st.cache_data(show_spinner=False)
def cached_download_youtube(link: str):
    return download_youtube_video(link)

@st.cache_data(show_spinner=False)
def cached_extract_audio(video_path: str):
    return extract_audio(video_path)

@st.cache_data(show_spinner=False)
def cached_transcribe_audio(audio_path: str):
    text, transcript_file = transcribe_audio_whisper(audio_path)
    return text

@st.cache_data(show_spinner=False)
def cached_generate_summary(text: str):
    return bart_summary(text)

# ========================
# 🔹 Initialize session state
# ========================
if "video_path" not in st.session_state:
    st.session_state.video_path = None

# ========================
# 🔹 User Input Section
# ========================
option = st.radio("Choose Input Type:", ["YouTube Link", "Upload Video"])

if option == "YouTube Link":
    youtube_link = st.text_input("Enter YouTube Video URL:")
    if youtube_link and st.button("Download Video"):
        with st.spinner("📥 Downloading YouTube video..."):
            st.session_state.video_path = cached_download_youtube(youtube_link)
        st.success("✅ YouTube video downloaded successfully!")

elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mkv", "mov"])
    if uploaded_video is not None:
        temp_path = uploaded_video.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.read())
        st.session_state.video_path = temp_path
        st.success("✅ Video uploaded successfully!")

# ========================
# 🔹 Process the Video
# ========================
if st.session_state.video_path and st.button("Generate Transcript and Summary"):
    progress = st.progress(0)

    # Step 1: Extract audio
    progress.progress(20)
    st.info("🎧 Extracting audio...")
    audio_path = cached_extract_audio(st.session_state.video_path)

    # Step 2: Transcribe
    progress.progress(60)
    st.info("🗣️ Transcribing audio using Whisper... (may take time)")
    transcript_text = cached_transcribe_audio(audio_path)
    progress.progress(80)
    st.success("✅ Transcript generated successfully!")

    # Show transcript
    st.subheader("📝 Transcript")
    st.text_area("Transcript", transcript_text[:3000], height=250)
    st.download_button("📥 Download Transcript", transcript_text, "transcript.txt")

    # Step 3: Summarize
    progress.progress(90)
    st.info("🧠 Generating summary...")
    summary = cached_generate_summary(transcript_text)
    progress.progress(100)
    st.success("✅ Summary generated successfully!")

    st.subheader("📃 Summary")
    st.write(summary)
    st.download_button("📥 Download Summary", summary, "summary.txt")

    # save_chunks_to_json(transcript_text)

