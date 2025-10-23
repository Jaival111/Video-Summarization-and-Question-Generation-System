
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from tqdm import tqdm
import re
import json
import os

# ================================
# CONFIGURATION
# ================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 350
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
def build_prompt(chunk_index: int, chunk_text: str, topic: str, n_subjective: int = 1, n_mcq: int = 1) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an educator creating exam-style questions from the following educational text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
CHUNK {chunk_index}:
{chunk_text}

Generate:
- 1 subjective question.
- 1 multiple-choice question (4 options + correct answer).
Make sure all questions are factually based on the text and clearly related to "{topic}".
Ensure that the subjective and multiple-choice questions are **different** from each other in focus and phrasing.
- Provide the chunk index as reference for each question.

Format your answer as:

### Subjective Questions:
1. ... (Reference: Chunk {chunk_index})

### Multiple Choice Questions:
1. ...
   A)
   B)
   C)
   D)
   Correct Answer: ... (Reference: Chunk {chunk_index})
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


# ================================
# TOPIC GENERATION FUNCTION
# ================================
def generate_topic(full_transcript: str, max_new_tokens=64):
    """Generate a concise topic/title for the given transcript."""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful teaching assistant.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Analyze the following lecture transcript and identify its main topic or title.

Transcript:
{full_transcript}

Respond with only one concise title or topic (max 10 words).
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )[0]["generated_text"].split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    
    # Extract just the title part (remove prompt text)
    # topic = clean_topic(output)
    return output


# ================================
# PARSING FUNCTIONS
# ================================
def parse_questions_from_output(output_text: str, chunk_index: int) -> dict:
    """Parse questions from a single output text."""
    questions = {
        "chunk_index": chunk_index,
        "subjective_question": "",
        "multiple_choice_question": "",
        "mcq_options": [],
        "correct_answer_index": -1
    }
    
    # Split by sections
    sections = output_text.split("###")
    
    for section in sections:
        section = section.strip()
        if "Subjective Questions:" in section:
            # Extract subjective question
            lines = section.split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("Subjective Questions:"):
                    # Remove numbering and reference
                    question = re.sub(r'^\d+\.\s*', '', line)
                    question = re.sub(r'\s*\(Reference: Chunk \d+\)', '', question)
                    if question:
                        questions["subjective_question"] = question
                        break
        
        elif "Multiple Choice Questions:" in section:
            # Extract multiple choice question
            lines = section.split("\n")
            current_question = ""
            options = []
            correct_answer = ""
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith("Multiple Choice Questions:"):
                    if re.match(r'^\d+\.', line):
                        # Question
                        current_question = re.sub(r'^\d+\.\s*', '', line)
                        current_question = re.sub(r'\s*\(Reference: Chunk \d+\)', '', current_question)
                    elif re.match(r'^[A-D]\)', line):
                        # Option
                        options.append(line)
                    elif line.startswith("Correct Answer:"):
                        # Correct answer
                        correct_answer = line.replace("Correct Answer:", "").strip()
                        correct_answer = re.sub(r'\s*\(Reference: Chunk \d+\)', '', correct_answer)
            
            questions["multiple_choice_question"] = current_question
            questions["mcq_options"] = options
            
            # Find correct answer index
            if correct_answer:
                for i, option in enumerate(options):
                    if option.strip() == correct_answer.strip():
                        questions["correct_answer_index"] = i
                        break
    
    return questions

def save_questions_to_json(questions_list: list, output_file: str = "generated_questions.json"):
    """Save questions to JSON file."""
    import datetime
    
    # Create the final structure
    output_data = {
        "metadata": {
            "generated_at": datetime.datetime.now().isoformat(),
            "model_used": MODEL_NAME,
            "device": DEVICE,
            "total_chunks": len(questions_list),
            "total_questions": len(questions_list) * 2  # 1 subjective + 1 MCQ per chunk
        },
        "questions": questions_list
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Questions saved to: {output_file}")

# ================================
# MAIN FUNCTION (batched & fast)
# ================================
@torch.inference_mode()
def generate_qna_with_references(df: pd.DataFrame, topic: str, n_subjective: int = 1, n_mcq: int = 1, save_json: bool = True):
    prompts = [
        build_prompt(idx, chunk_text, topic, n_subjective, n_mcq)
        for idx, chunk_text in df.itertuples(index=False)
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
    
    if save_json:
        # Parse each output and create structured data
        questions_list = []
        for idx, output in enumerate(all_outputs):
            parsed_questions = parse_questions_from_output(output, idx + 1)  # chunk_index starts from 1
            questions_list.append(parsed_questions)
        
        # Save to JSON
        save_questions_to_json(questions_list)
        return questions_list
    else:
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

    # Generate questions and save as JSON
    questions_data = generate_qna_with_references(
        text_df, 
        topic="Building a Blog Application with Flask: Features and Setup.", 
        n_subjective=1, 
        n_mcq=1,
        save_json=True
    )
    
    print(f"\nGenerated questions for {len(questions_data)} chunks")
    print("Questions saved to 'generated_questions.json'")
    
    # Display a preview of the JSON structure
    print("\nJSON Structure Preview:")
    if questions_data:
        sample = questions_data[0]
        print(json.dumps({
            "chunk_index": sample["chunk_index"],
            "subjective_question": sample["subjective_question"],
            "multiple_choice_question": sample["multiple_choice_question"],
            "mcq_options": sample["mcq_options"],
            "correct_answer_index": sample["correct_answer_index"]
        }, indent=2))
