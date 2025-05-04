import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the pretrained VGG16 model
model = VGG16(weights='imagenet')

# Load and preprocess image
img_path = 'example.jpg'  # Change this to your image path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)

# Predict using VGG16
preds = model.predict(img_preprocessed)

# Decode predictions and display top 3
top_preds = decode_predictions(preds, top=3)[0]

# Display image and predictions
plt.imshow(img)
plt.axis('off')
plt.title("Top Predictions:")
plt.show()

for i, (imagenet_id, label, prob) in enumerate(top_preds):
    print(f"{i+1}. {label}: {prob * 100:.2f}%")
