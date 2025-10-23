from transformers import pipeline
import textwrap
import os
from preprocess import chunk_text

def summarize_text(text, model_name="facebook/bart-large-cnn", output_file="summary.txt"):
    """Generate a coherent summary for the whole text."""
    summarizer = pipeline("summarization", model=model_name, device_map="auto")

    # Step 1: Split the text
    chunks = chunk_text(text)
    print(f"🔹 Total chunks created: {len(chunks)}")

    # Step 2: Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")
        # Handle both string and dictionary formats
        if isinstance(chunk, dict):
            chunk_content = chunk.get("chunk_text", "")
        else:
            chunk_content = chunk

        summary = summarizer(chunk_content, max_length=30, min_length=10, do_sample=False)[0]["summary_text"]
        summaries.append(summary)

    # Step 3: Combine summaries and re-summarize to get a clean final summary
    combined_summary = " ".join(summaries)
    # final_summary = summarizer(combined_summary, max_length=300, min_length=120, do_sample=False)[0]["summary_text"]
    
    # Write summary to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_summary)
    
    print(f"Summary saved to: {output_file}")
    return combined_summary, output_file

def bart_summary(text):
    """Wrapper function for backward compatibility with main.py"""
    summary, _ = summarize_text(text)
    return summary


if __name__ == "__main__":
    # Path to your transcript
    transcript_path = "transcript.txt"

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    print("Generating summary...")
    final_summary, summary_file = summarize_text(transcript)

    print(f"\n✅ Summary saved to '{summary_file}'\n")
    print(textwrap.fill(final_summary, width=100))
