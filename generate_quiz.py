
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from tqdm import tqdm
import re

# ================================
# CONFIGURATION
# ================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 300
MAX_NEW_TOKENS = 350  # reduce from 512
BATCH_SIZE = 4

# ================================
# LOAD MODEL (optimized)
# ================================
print(f"Loading model '{MODEL_NAME}' on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    batch_size=BATCH_SIZE
)

# set padding correctly
generator.tokenizer.pad_token = generator.tokenizer.eos_token
generator.model.config.pad_token_id = generator.model.config.eos_token_id


# def clean_output(text: str) -> str:
#     matches = list(re.finditer(r"###\s*Subjective Questions", text))
#     if matches:
#         last_match = matches[-1]
#         return text[last_match.start():].strip()
#     return text.strip()

# def clean_topic(text: str) -> str:
#     matches = list(re.finditer(r"(?<=max 10 words\.)\s*(.*)", text))
#     if matches:
#         last_result = matches[-1].group(1).strip()
#         return last_result
#     return text.strip()


# ================================
# PROMPT BUILDING
# ================================
def build_prompt(chunk_text: str, topic: str, n_subjective: int = 1, n_mcq: int = 1) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an educator creating exam-style questions from the following educational text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Text:
{chunk_text}

Generate:
- 1 subjective question.
- 1 multiple-choice question (4 options + correct answer).
Make sure all questions are factually based on the text and clearly related to "{topic}".

Format your answer as:

### Subjective Questions:
1. ...

### Multiple Choice Questions:
1. ...
   A)
   B)
   C)
   D)
   Correct Answer: ...
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


# ================================
# TOPIC GENERATION FUNCTION
# ================================
# def generate_topic(full_transcript: str, max_new_tokens=64):
#     """Generate a concise topic/title for the given transcript."""
#     prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful teaching assistant.
# <|eot_id|><|start_header_id|>user<|end_header_id|>
# Analyze the following lecture transcript and identify its main topic or title.

# Transcript:
# {full_transcript}

# Respond with only one concise title or topic (max 10 words).
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """
#     output = generator(
#         prompt,
#         max_new_tokens=max_new_tokens,
#         temperature=0.3,
#         do_sample=False,
#         pad_token_id=generator.tokenizer.eos_token_id
#     )[0]["generated_text"].split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    
#     # Extract just the title part (remove prompt text)
#     # topic = clean_topic(output)
#     return output


# ================================
# MAIN FUNCTION (batched & fast)
# ================================
@torch.inference_mode()
def generate_qna_with_references(df: pd.DataFrame, topic: str, n_subjective: int = 1, n_mcq: int = 1):
    prompts = [
        build_prompt(chunk_text, topic, n_subjective, n_mcq)
        for _, chunk_text in df.itertuples(index=False)
    ]

    all_outputs = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating QnA"):
        batch = prompts[i:i + BATCH_SIZE]
        outputs = generator(
            batch,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.6,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        all_outputs.extend(o[0]['generated_text'].split("<|start_header_id|>assistant<|end_header_id|>")[-1] for o in outputs)

    return "\n\n".join(all_outputs)

# ================================
# USAGE
# ================================
if __name__ == "__main__":
    text_df = pd.read_json("chunks_output/text_chunks.json")
    # with open("transcripts/transcript_1.txt", "r", encoding="utf-8") as f:
    #     transcript = f.read()
    # print("Generating topic...")
    # topic = generate_topic(transcript, generator)
    # print(f"Identified Topic: {topic}\n")

    qna_with_refs = generate_qna_with_references(text_df, topic="Building a Blog Application with Flask: Features and Setup.", n_subjective=1, n_mcq=1)
    print(qna_with_refs)
