# Measuring Diversity in AI-Generated Data – Develop metrics to measure diversity and novelty in AI-generated images or text.

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math

# Load the InceptionV3 model without the top layer (pre-trained feature extractor)
inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
inception_model.trainable = False

# Function to calculate pairwise cosine distances between generated images
def calculate_image_diversity(images):
    # Resize images to the size InceptionV3 expects (299x299)
    resized_images = np.array([tf.image.resize(image, (299, 299)) for image in images])
    resized_images = np.expand_dims(resized_images, axis=0)

    # Get feature representations using the Inception model
    features = inception_model.predict(resized_images)

    # Calculate pairwise cosine distances
    cosine_sim = cosine_similarity(features)
    diversity_score = np.mean(cosine_sim)  # Lower means more diverse
    return diversity_score

# Function to calculate distinct n-grams (bigrams)
def distinct_ngrams(texts, n=2):
    ngrams = []
    for text in texts:
        words = text.split()
        ngrams += [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    ngram_counts = Counter(ngrams)
    distinct_ngrams = len(ngram_counts)
    total_ngrams = len(ngrams)
    return distinct_ngrams / total_ngrams  # The ratio of unique n-grams

# Function to calculate entropy of generated text
def calculate_entropy(texts):
    text = ' '.join(texts)
    text_length = len(text)
    char_freq = Counter(text)
    entropy = -sum((freq / text_length) * math.log2(freq / text_length) for freq in char_freq.values())
    return entropy

# Example generated images (replace with real generated images)
generated_images = np.random.rand(100, 32, 32, 3)  # Replace with actual image data
# Calculate image diversity score
image_diversity_score = calculate_image_diversity(generated_images)
print(f"Image Diversity score (lower is more diverse): {image_diversity_score}")

# Example generated texts
generated_texts = [
    "This is a test sentence.",
    "The quick brown fox jumps over the lazy dog.",
    "Another example sentence to check diversity."
]

# Calculate distinct bigrams and entropy for text
distinct_bigrams = distinct_ngrams(generated_texts, n=2)
text_entropy = calculate_entropy(generated_texts)

print(f"Distinct bigrams ratio: {distinct_bigrams}")
print(f"Entropy of generated text: {text_entropy}")
