# Text Generation Using LSTM or Transformer Models – Train an AI model to generate poetry, news articles, or dialogue.

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- Load Pretrained GPT-2 ---
model_name = "gpt2"  # or use "gpt2-medium", "gpt2-large" for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# --- Use GPU if available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Text Generation Function ---
def generate_text(prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Prompt Examples ---
prompts = {
    "poetry": "In the moonlight, I saw",
    "news": "Breaking news from the capital city",
    "dialogue": "AI: Hello there! Human:"
}

# --- Generate Samples ---
for category, prompt in prompts.items():
    print(f"\n--- {category.capitalize()} ---")
    print(generate_text(prompt))
