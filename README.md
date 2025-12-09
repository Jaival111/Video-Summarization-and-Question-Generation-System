# 🎥 Audio & Video Transcripts, Summaries & Question Generation System

An end-to-end AI pipeline for transforming educational video/audio content into transcripts, summaries, and quizzes

## 📌 Overview

This project provides a fully automated pipeline that converts raw video/audio content into structured educational material.

It supports YouTube links, local uploads, and multiple media formats, generating:

- High-accuracy transcripts
- Abstractive summaries
- Educational question–answer pairs (MCQs + descriptive)

Built using state-of-the-art speech-to-text and LLM models, the system is designed for corporate training, online learning platforms, and scalable educational content creation.

## 🎯 Features

- Audio extraction from video sources
- Speech-to-text transcription using Whisper
- Abstractive summarization using BART
- Automatic QnA generation using Llama models
- Chunked & modular pipeline for large files
- Relevance filtering to remove hallucinations
- Streamlit interface for easy uploads and results visualisation
- Extensible architecture—supports fine-tuning, additional models, and cloud scaling

## 🖼️ System Architecture

<img src="pipeline.png">

## 🛠️ Technologies & Tools Used

| Components     | Technology    |
|----------------|---------------|
| Video Download | yt-dlp |
| Audio Extraction | FFmpeg |
| Speech-to-Text | OpenAI Whisper |
| Summarization | BART (facebook/bart-large-cnn) |
| Question Generation | Llama 3.1 8B (base + fine-tuned) |
| Web UI | Streamlit |
| Model Training | Transformers, PyTorch |

## 🚀 Workflow

### 1. Input Handling
    Upload a local file or provide a YouTube URL.
    → Processed using yt-dlp.

### 2. Audio Extraction
    Extracted using FFmpeg for high-quality STT input.

### 3. Transcription
    Converted to text using OpenAI Whisper, with timestamp alignment.

### 4. Summarization
    Each transcript chunk is summarized using facebook/bart-large-cnn.

### 5. Question Generation
    Using Llama 3.1 8B (base + fine-tuned):
    - Generates descriptive + MCQ questions
    - Performs relevance verification to avoid hallucinations
    - Retains only high-confidence QnA pairs

### 6. User Interface
    A Streamlit UI displays:
    - Transcript
    - Summary
    - Generated questions
    - Downloadable outputs

## 📈 Evaluation

### Whisper achieved ~99% transcription accuracy on test videos.

### The fine-tuned Llama model produced higher-quality, context-aligned QnAs compared to the base model.

### Relevance filtering significantly reduced hallucinations, improving reliability.

## 🧭 Future Enhancements

- Add question types: fill-in-the-blanks, true/false, match-the-following

- Integrate human-in-the-loop active learning

- Deploy with cloud GPUs for faster batch processing

- Add multimodal question generation (image/audio-based)

## Outcomes

- Automated complete pipeline for transcription → summarization → question generation

- Reduced manual effort for creating educational material

- Developed a user-friendly interface

- Demonstrated strong performance and commercial viability

## 📚 References

- OpenBMB, RAGEval Dataset Generation Framework

- Research on Whisper-based transcription

- Llama & BART model documentation

## 🔗 Useful Links

### [Kaggle Model (Download Full Pipeline Model)](https://www.kaggle.com/models/vikram1213/video-summarization-and-quiz-generation)

### [GitHub Repository](https://github.com/Jaival111/Video-Summarization-and-Question-Generation-System)
