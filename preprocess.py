import os
import json
from transformers import AutoTokenizer

# ================================
# CONFIGURATION
# ================================
with open("transcript.txt", "r", encoding="utf-8") as f:
    INPUT_TEXT = f.read()
OUTPUT_DIR = "chunks_output"
CHUNK_SIZE = 50
OUTPUT_JSON_FILE = "text_chunks.json"

# ================================
# HELPER FUNCTION TO SPLIT TEXT
# ================================
def chunk_text(
    text,
    tokenizer=None,
    max_tokens=128
):
    """Split text into chunks that respect sentence boundaries and token limits."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            use_fast=True
        )

    # Split by sentence (naively by '. ') — you can improve using nltk or spacy
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        # Ensure sentence ends with a period
        if not sentence.endswith('.'):
            sentence += '.'

        # Tokenize the sentence
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        n_tokens = len(sentence_tokens)

        # If adding this sentence exceeds limit, save current chunk
        if current_tokens + n_tokens > max_tokens:
            if current_chunk.strip():
                chunks.append({
                    "chunk_index": len(chunks) + 1,
                    "chunk_text": current_chunk.strip()
                })
            # start a new chunk with the current sentence
            current_chunk = sentence + " "
            current_tokens = n_tokens
        else:
            current_chunk += sentence + " "
            current_tokens += n_tokens

    # Add the last chunk
    if current_chunk.strip():
        chunks.append({
            "chunk_index": len(chunks) + 1,
            "chunk_text": current_chunk.strip()
        })

    return chunks


# ================================
# MAIN FUNCTION
# ================================
def save_chunks_to_json(text, output_dir=OUTPUT_DIR, output_file=OUTPUT_JSON_FILE, chunk_size=CHUNK_SIZE):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    chunks = chunk_text(text=text)
    
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to {output_path}")

# ================================
# RUN SCRIPT
# ================================
if __name__ == "__main__":
    save_chunks_to_json(INPUT_TEXT)