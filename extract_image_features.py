# extract_image_features.py

import os
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tqdm import tqdm
import pickle

# Load InceptionV3 model (remove last layer)
def load_model():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)  # 2048D
    return model

# Extract features from a single image
def extract_features_single(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()

# Extract features from all images in a directory
def extract_features_bulk(image_dir, image_ids):
    model = load_model()
    features = {}
    for img_id in tqdm(image_ids):
        img_path = os.path.join(image_dir, img_id)
        if os.path.exists(img_path):
            features[img_id] = extract_features_single(img_path, model)
    return features

# Save to .pkl
def save_features(features, filename='image_features.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(features, f)


if __name__ == "__main__":
    image_dir = "C:/Users/swapn/Downloads/archive/Images"  # or wherever your images are
    image_ids = os.listdir(image_dir)
    features = extract_features_bulk(image_dir, image_ids)
    save_features(features, filename="image_features.pkl")  
 