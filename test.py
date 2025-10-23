import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # ✅ Chat-tuned version
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model '{MODEL_NAME}' on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# set padding correctly
generator.tokenizer.pad_token = generator.tokenizer.eos_token
generator.model.config.pad_token_id = generator.model.config.eos_token_id

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an educator creating exam-style questions from the following educational text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Text:
Now running the module directly used to be how I always ran flask applications, but now the flask documentation uses the flask run command. So I've been using that as well. So the flask command with the environment variables also allows us to use the flask shell for some debugging. And we'll see a couple of examples of that in later videos. Now in this series, I probably will be running the application directly with Python just because I don't want to keep setting those environment variables again whenever I shut down my terminal.

Generate:
- 1 subjective question.
- 1 multiple-choice question (4 options + correct answer).
Make sure all questions are factually based on the text and clearly related to "Building a Blog Application with Flask: Features and Setup.".

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

output = generator(
    prompt,
    max_new_tokens=350,
    temperature=0.6,
    do_sample=True,
    pad_token_id=generator.tokenizer.eos_token_id
)

print("\nGenerated Output:\n")
print(output[0]['generated_text'].split("<|start_header_id|>assistant<|end_header_id|>")[-1])
print(output)