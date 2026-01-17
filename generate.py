import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

model_name = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model.eval()

MCQ_PROMPT = """
You are an expert educator.

Given the following video transcript, generate {num_questions} high-quality multiple-choice questions (MCQs).

Rules:
- Questions must be directly answerable from the transcript
- Avoid vague or generic questions
- Each question must have exactly 4 options (A, B, C, D)
- Only ONE correct answer
- Provide the correct option and a short explanation

Transcript:
{transcript}

Return output strictly in JSON format as:
[
  {{
    "question": "...",
    "options": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "correct_answer": "A",
    "explanation": "..."
  }}
]
"""

def generate_mcqs(transcript_chunk, num_questions=5, max_tokens=1024):

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert educator. "
                "You MUST output ONLY valid JSON. "
                "Do NOT repeat the prompt. "
                "Generate EXACTLY the number of questions requested."
            )
        },
        {
            "role": "user",
            "content": MCQ_PROMPT.format(
                transcript=transcript_chunk,
                num_questions=num_questions
            )
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in decoded:
        decoded = decoded.split("assistant")[-1].strip()

    return decoded

def parse_mcqs(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("JSON parsing failed")
        return []


with open("chunks.json") as f:
    chunks = json.load(f)

all_mcqs = []

for chunk in chunks:
    response = generate_mcqs(chunk, num_questions=5)
    mcqs = parse_mcqs(response)
    all_mcqs.extend(mcqs)

with open("questions.json", "a") as f:
    json.dump(all_mcqs, f, indent=4)

