import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

EVAL_PROMPT = """
You are a strict evaluator.

Given:
1. A video transcript
2. A multiple-choice question generated from it

Decide whether the question is USEFUL and RELEVANT.

Criteria:
- Question must be directly based on the transcript
- Answer must be clearly supported by the transcript
- No hallucinated facts
- No trivial or off-topic questions

Transcript:
{transcript}

Question:
{question}

Options:
{options}

Correct Answer:
{correct_answer}

Respond ONLY in JSON:
{{
  "verdict": "YES" or "NO",
  "reason": "short explanation"
}}
"""

def evaluate_mcq(transcript_chunk, mcq):
    options_text = "\n".join([f"{k}: {v}" for k, v in mcq["options"].items()])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict evaluator. "
                "You MUST output ONLY valid JSON. "
                "Do not repeat the prompt."
            )
        },
        {
            "role": "user",
            "content": EVAL_PROMPT.format(
                transcript=transcript_chunk,
                question=mcq["question"],
                options=options_text,
                correct_answer=mcq["correct_answer"]
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
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    try:
        verdict_json = json.loads(response)
        return verdict_json
    except:
        return {"verdict": "NO", "reason": "Invalid JSON"}

with open("chunks.json") as f:
    transcript_chunks = json.load(f)

with open("questions.json") as f:
    mcqs = json.load(f)

verdicts = []

counter = 1
for chunk in transcript_chunks:
    current_chunk_mcqs = mcqs[5*(counter-1):5*counter]
    counter += 1
    for mcq in current_chunk_mcqs:
        result = evaluate_mcq(chunk, mcq)
        verdicts.append(result)

with open("verdicts.json", "a") as f:
    json.dump(verdicts, f, indent=4)