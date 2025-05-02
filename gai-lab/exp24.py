# Generative AI for Personalized Marketing – Build a chatbot that generates personalized marketing messages.

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Load GPT-2 or a fine-tuned marketing model from HuggingFace
model_name = "gpt2"  # Replace with your fine-tuned model if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Sample user profiles
customer_profiles = {
    "user123": {
        "name": "Alex",
        "age": 28,
        "interests": ["fitness", "tech gadgets", "travel"],
        "location": "New York",
        "purchases": ["smartwatch", "yoga mat"]
    },
    "user456": {
        "name": "Maria",
        "age": 35,
        "interests": ["fashion", "beauty products", "online courses"],
        "location": "Los Angeles",
        "purchases": ["skincare kit", "digital marketing course"]
    }
}

# Prompt engineering
def generate_prompt(profile):
    return (
        f"Create a personalized marketing message for:\n"
        f"Name: {profile['name']}\n"
        f"Age: {profile['age']}\n"
        f"Location: {profile['location']}\n"
        f"Interests: {', '.join(profile['interests'])}\n"
        f"Previous Purchases: {', '.join(profile['purchases'])}\n\n"
        f"Message:"
    )

@app.route("/message/<user_id>", methods=["GET"])
def personalized_message(user_id):
    if user_id in customer_profiles:
        profile = customer_profiles[user_id]
        prompt = generate_prompt(profile)
        output = generator(prompt, max_length=150, do_sample=True, temperature=0.9, top_p=0.95)[0]['generated_text']
        message = output.split("Message:")[-1].strip().split("\n")[0]  # Extract first message only
        return jsonify({"message": message})
    else:
        return jsonify({"error": "User not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
