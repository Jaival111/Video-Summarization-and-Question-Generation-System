import json

def filter_mcqs(verdicts, mcqs):
    accepted = []

    for verdict, mcq in zip(verdicts, mcqs):
        if verdict["verdict"] == "YES":
            accepted.append(mcq)

    return accepted

def format_for_lora(chunk, mcq):
    return {
        "instruction": "Generate a multiple-choice question from the transcript.",
        "input": chunk,
        "output": json.dumps(mcq, ensure_ascii=False)
    }

with open("chunks.json", "r") as f:
    chunks = json.load(f)

with open("questions.json", "r") as f:
    all_mcqs = json.load(f)

with open("verdicts.json", "r") as f:
    all_verdicts = json.load(f)

all_data = []

counter = 1
for chunk in chunks:
    mcqs = all_mcqs[5*(counter-1):5*counter]
    verdicts = all_verdicts[5*(counter-1):5*counter]

    accepted_mcqs = filter_mcqs(verdicts, mcqs)

    for mcq in accepted_mcqs:
        formatted = format_for_lora(chunk, mcq)
        all_data.append(formatted)

    counter += 1

with open("lora_dataset.jsonl", "w") as f:
    for item in all_data:
        f.write(json.dumps(item) + "\n")