import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import os
from scipy.spatial import distance

# Load VGG16 model + higher level layers
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data

def extract_features(img_data):
    features = model.predict(img_data)
    return features.flatten()

def load_database_images(database_dir):
    database_features = []
    image_paths = []

    for img_file in os.listdir(database_dir):
        img_path = os.path.join(database_dir, img_file)
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_data = preprocess_image(img_path)
            features = extract_features(img_data)
            database_features.append(features)
            image_paths.append(img_path)

    return np.array(database_features), image_paths

def find_most_similar(target_features, database_features):
    distances = distance.cdist([target_features], database_features, 'euclidean')
    min_index = np.argmin(distances)
    return min_index, distances[0][min_index]

def calculate_similarity_percentage(distance, max_distance):
    # Convert the distance to a percentage of similarity
    similarity_percentage = max(0, 100 - (distance / max_distance * 100))
    return similarity_percentage

def display_image(image_path, similarity_percentage):
    window = tk.Toplevel()
    window.title("Most Similar Image")

    img = Image.open(image_path)
    img = img.resize((300, 300), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)

    label_img = tk.Label(window, image=img_tk)
    label_img.image = img_tk  # Keep a reference to avoid garbage collection
    label_img.pack()

    label_text = tk.Label(window, text=f"Similarity: {similarity_percentage:.2f}%")
    label_text.pack()

    window.mainloop()

# Set up the main GUI
root = tk.Tk()
root.withdraw()  # Hide the root window

# Ask the user to select the target image
target_img_path = filedialog.askopenfilename(title='Select Target Image', filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
if not target_img_path:
    print("No image selected")
    exit()

# Ask the user to select the database directory
database_dir = filedialog.askdirectory(title='Select Database Directory')
if not database_dir:
    print("No directory selected")
    exit()

# Preprocess the target image and extract features
target_img_data = preprocess_image(target_img_path)
target_features = extract_features(target_img_data)

# Load and preprocess database images
database_features, image_paths = load_database_images(database_dir)

# Find the most similar image in the database
most_similar_index, distance_to_most_similar = find_most_similar(target_features, database_features)
most_similar_image_path = image_paths[most_similar_index]

# Assume a reasonable max_distance for normalization
max_distance = np.max(distance.cdist([target_features], database_features, 'euclidean'))

# Calculate similarity percentage
similarity_percentage = calculate_similarity_percentage(distance_to_most_similar, max_distance)

print(f"Most similar image: {most_similar_image_path}")
print(f"Similarity: {similarity_percentage:.2f}%")

# Display the most similar image and similarity percentage in a new GUI window
display_image(most_similar_image_path, similarity_percentage)
