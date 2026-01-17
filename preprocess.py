import os
import json
from transformers import AutoTokenizer
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

with open("transcript.txt", "r", encoding="utf-8") as f:
    INPUT_TEXT = f.read()
OUTPUT_DIR = "chunks_output"
CHUNK_SIZE = 50
OUTPUT_JSON_FILE = "text_chunks.json"

def chunk_text(
    text,
    tokenizer_name="facebook/bart-large-cnn",
    max_tokens=900,   # leave buffer for BART
    overlap_tokens=100
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(
            sentence,
            add_special_tokens=False
        )
        n_tokens = len(sentence_tokens)

        # Handle very long sentences
        if n_tokens > max_tokens:
            sentence_tokens = sentence_tokens[:max_tokens]
            sentence = tokenizer.decode(sentence_tokens)

        if current_tokens + n_tokens > max_tokens:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # overlap
            overlap = tokenizer.encode(
                chunk_text,
                add_special_tokens=False
            )[-overlap_tokens:]
            current_chunk = [tokenizer.decode(overlap)]
            current_tokens = len(overlap)

        current_chunk.append(sentence)
        current_tokens += n_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def save_chunks_to_json(text, output_dir=OUTPUT_DIR, output_file=OUTPUT_JSON_FILE, chunk_size=CHUNK_SIZE):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    chunks = chunk_text(text=text)
    
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to {output_path}")

if __name__ == "__main__":
    save_chunks_to_json(INPUT_TEXT)