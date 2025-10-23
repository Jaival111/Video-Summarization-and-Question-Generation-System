import streamlit as st
import os
import json
import pandas as pd
from extract_audio import download_youtube_video, extract_audio
from audio_to_text import transcribe_audio_whisper
from summarize import bart_summary
from preprocess import save_chunks_to_json, chunk_text
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

@st.cache_data(show_spinner=False)
def cached_preprocess_transcript(text: str):
    """Preprocess transcript into chunks."""
    chunks = chunk_text(text)
    return chunks

@st.cache_data(show_spinner=False)
def cached_generate_quiz(chunks, topic: str):
    """Generate quiz questions from chunks."""
    # Convert chunks to DataFrame format expected by generate_quiz
    df = pd.DataFrame(chunks)
    questions_data = generate_qna_with_references(df, topic, save_json=True)
    return questions_data

# ========================
# 🔹 Initialize session state
# ========================
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "questions_data" not in st.session_state:
    st.session_state.questions_data = None

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
    progress.progress(40)
    st.info("🗣️ Transcribing audio using Whisper... (may take time)")
    transcript_text = cached_transcribe_audio(audio_path)
    st.session_state.transcript_text = transcript_text
    progress.progress(60)
    st.success("✅ Transcript generated successfully!")

    # Show transcript
    st.subheader("📝 Transcript")
    st.text_area("Transcript", transcript_text[:3000], height=250)
    st.download_button("📥 Download Transcript", transcript_text, "transcript.txt")

    # Step 3: Preprocess into chunks
    progress.progress(70)
    st.info("🔧 Preprocessing transcript into chunks...")
    chunks = cached_preprocess_transcript(transcript_text)
    st.session_state.chunks = chunks
    progress.progress(80)
    st.success(f"✅ Preprocessed into {len(chunks)} chunks!")

    # Step 4: Generate topic
    progress.progress(85)
    st.info("🎯 Generating topic...")
    topic = generate_topic(transcript_text)
    progress.progress(90)
    st.success(f"✅ Topic identified: {topic}")

    # Step 5: Generate quiz questions
    progress.progress(95)
    st.info("❓ Generating quiz questions...")
    questions_data = cached_generate_quiz(chunks, topic)
    st.session_state.questions_data = questions_data
    progress.progress(100)
    st.success(f"✅ Generated {len(questions_data)} question sets!")

    # Step 6: Summarize
    st.info("🧠 Generating summary...")
    summary = cached_generate_summary(transcript_text)
    st.success("✅ Summary generated successfully!")

    st.subheader("📃 Summary")
    st.write(summary)
    st.download_button("📥 Download Summary", summary, "summary.txt")

# ========================
# 🔹 Display Chunks (Optional)
# ========================
if st.session_state.chunks:
    with st.expander("🔍 View Text Chunks"):
        st.write("**Text Chunks for Quiz Generation:**")
        for i, chunk in enumerate(st.session_state.chunks):
            st.write(f"**Chunk {chunk['chunk_index']}:**")
            st.write(chunk['chunk_text'][:200] + "..." if len(chunk['chunk_text']) > 200 else chunk['chunk_text'])
            st.divider()

# ========================
# 🔹 Display Generated Questions
# ========================
if st.session_state.questions_data:
    st.subheader("❓ Generated Quiz Questions")
    
    # Display questions in tabs
    tab1, tab2 = st.tabs(["📝 Subjective Questions", "🔘 Multiple Choice Questions"])
    
    with tab1:
        st.write("**Subjective Questions:**")
        for i, question_data in enumerate(st.session_state.questions_data):
            if question_data["subjective_question"]:
                st.write(f"**Chunk {question_data['chunk_index']}:** {question_data['subjective_question']}")
                st.divider()
    
    with tab2:
        st.write("**Multiple Choice Questions:**")
        for i, question_data in enumerate(st.session_state.questions_data):
            if question_data["multiple_choice_question"]:
                st.write(f"**Chunk {question_data['chunk_index']}:** {question_data['multiple_choice_question']}")
                
                # Display options
                for j, option in enumerate(question_data["mcq_options"]):
                    if j == question_data["correct_answer_index"]:
                        st.write(f"✅ {option} (Correct)")
                    else:
                        st.write(f"   {option}")
                
                st.divider()
    
    # Download questions as JSON
    questions_json = json.dumps({
        "metadata": {
            "total_chunks": len(st.session_state.questions_data),
            "total_questions": len(st.session_state.questions_data) * 2
        },
        "questions": st.session_state.questions_data
    }, indent=2)
    
    st.download_button(
        "📥 Download Questions (JSON)", 
        questions_json, 
        "generated_questions.json", 
        mime="application/json"
    )

