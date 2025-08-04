This deep learning project generates descriptive captions for images using a combination of Convolutional Neural Networks (CNNs) for image feature extraction and Long Short-Term Memory (LSTM) networks for natural language generation.

📦 Dataset used: Flickr8k Dataset from Kaggle : https://www.kaggle.com/datasets/adityajn105/flickr8k

🤖 Model: Encoder-Decoder architecture with CNN + LSTM

🧠 Workflow Overview

build_captions_dict.py

Reads raw caption data and builds a dictionary mapping image IDs to cleaned caption lists.

🧑‍🏭build_tokenizer.py

Creates a tokenizer from all captions and saves it as a pickle file.

🌌extract_image_features.py

Uses a pretrained CNN (e.g., InceptionV3 or VGG16) to extract image feature vectors.

🤖prepare_sequences.py

Converts captions into padded sequences suitable for training and saves X1, X2, and y.

🤖train_model.py

Trains the model using image features and caption sequences, saving it as model_caption.keras.


🧑‍🏭generate_caption.py

Loads the model and generates captions for new images using greedy or beam search decoding.

📊Bleu.py

Evaluates the model using BLEU scores to measure translation quality.

🔧 Tools & Libraries
TensorFlow / Keras

NumPy

Pickle

InceptionV3

LSTM

Python's os / glob modules

BLEU Score (NLTK or custom implementation)

