import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model_name = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,  # rank
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from datasets import load_dataset
dataset = load_dataset("json", data_files="train_qlora_dataset.jsonl")["train"]

def tokenize_function(example):
    text = f"{example['prompt']}{example['response']}"
    tokenized = tokenizer(
        text,
        max_length=2048,
        truncation=True,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function)

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./llama3-qlora-qna",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    learning_rate=2e-4,
    max_steps=1000,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("./llama3-qna-qlora-adapter")

from transformers import pipeline

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an educator creating exam-style questions from the following educational text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Text:
Flask allows developers to build lightweight, modular web applications quickly.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

output = generator(prompt, max_new_tokens=300, temperature=0.7, do_sample=True)[0]["generated_text"]
print(output)
