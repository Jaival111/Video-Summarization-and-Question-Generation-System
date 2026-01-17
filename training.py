
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

dataset = load_dataset(
    "json",
    data_files="lora_dataset.jsonl"
)

train_dataset = dataset["train"]

model_name = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
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

def format_example(example):
    messages = [
        {
            "role": "system",
            "content": "You are an expert educator who generates high-quality MCQs."
        },
        {
            "role": "user",
            "content": f"{example['instruction']}\n\nTranscript:\n{example['input']}"
        },
        {
            "role": "assistant",
            "content": example["output"]
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=1024,
    )

    return tokenizer.decode(tokens["input_ids"])

training_args = TrainingArguments(
    output_dir="./lora-mcq-llama3-new",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    optim="paged_adamw_8bit",
    report_to="none"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.config.use_cache = False

model.train()

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    formatting_func=format_example,
    processing_class=tokenizer
)

trainer.train()

trainer.model.save_pretrained("./lora-mcq-llama3-new")
tokenizer.save_pretrained("./lora-mcq-llama3-new")

