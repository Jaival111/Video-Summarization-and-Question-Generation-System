import json
import pandas as pd

# Load text chunks
df = pd.read_json("chunks_output/text_chunks.json")

# Load generated questions
with open("generated_questions.json", "r", encoding="utf-8") as f:
    questions_data = json.load(f)["questions"]

train_data = []
for i, (idx, row) in enumerate(df.iterrows()):
    if i >= len(questions_data):
        continue
    
    chunk_text = row['chunk_text']
    topic = "Building a Blog Application with Flask: Features and Setup."
    q = questions_data[i]
    
    # Create input prompt
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an educator creating exam-style questions from the following educational text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Text:
{chunk_text}

Generate:
- 1 subjective question.
- 1 multiple-choice question (4 options + correct answer).
Make sure all questions are factually based on the text and clearly related to "{topic}".
Ensure that the subjective and multiple-choice questions are **different** from each other.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    # Combine the generated outputs
    # Combine the generated outputs
    options_text = "\n   ".join(q['mcq_options'])
    output = (
        f"### Subjective Questions:\n"
        f"1. {q['subjective_question']}\n\n"
        f"### Multiple Choice Questions:\n"
        f"1. {q['multiple_choice_question']}\n"
        f"   {options_text}\n"
        f"   Correct Answer: {q['mcq_options'][q['correct_answer_index']]}"
    )


    train_data.append({"prompt": prompt, "response": output})

pd.DataFrame(train_data).to_json("train_qlora_dataset.jsonl", orient="records", lines=True)
print(f"✅ Saved {len(train_data)} training examples.")
