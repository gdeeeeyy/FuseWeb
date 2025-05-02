# Fine-Tuning GPT for Custom Text Generation – Train GPT on domain-specific data to generate high-quality text.

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import torch

# --- Load or Create Custom Dataset ---
# Replace this list with your domain-specific text
custom_texts = [
    "Quantum computing is the future of secure communication.",
    "In the realm of medicine, AI can diagnose diseases faster than humans.",
    "Poetry speaks to the soul when structured with rhythm and metaphor.",
]

# Convert to Hugging Face Dataset format
dataset = Dataset.from_dict({"text": custom_texts})

# --- Load Tokenizer and Model ---
model_name = "gpt2"  # or "gpt2-medium", "gpt2-small"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token

model = GPT2LMHeadModel.from_pretrained(model_name)

# --- Tokenize Dataset ---
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=10,
    logging_dir="./logs",
)

# --- Data Collator for Language Modeling ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- Fine-tune ---
trainer.train()

# --- Text Generation with Fine-tuned Model ---
model.eval()
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example prompt after training
print(generate("In the realm of"))
