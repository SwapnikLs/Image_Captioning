from keras.models import load_model
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from PIL import Image
import matplotlib.pyplot as plt
from prepare_sequences import max_length

# Load max_length from your prepare_sequences.py (or set explicitly if necessary)
max_lengt= max_length  # <-- Replace with value from your data pipeline! Or: from prepare_sequences import max_length

# Load model
model = load_model('model_caption.keras', compile=False)

# Load tokenizer
with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Reverse word map
index_word = {v: k for k, v in tokenizer.word_index.items()}

def generate_caption(photo, max_len=max_lengt):
    in_text = 'startseq'
    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        yhat = model.predict([photo, seq], verbose=0)
        predicted_id = np.argmax(yhat)
        word = index_word.get(predicted_id)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    # Remove startseq and endseq, then return
    return ' '.join(in_text.split()[1:-1])

def display_image_with_caption(image_path, caption):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption, fontsize=12)
    plt.show()

# Load pre-extracted image features
with open('data/image_features.pkl', 'rb') as f:
    image_features = pickle.load(f)

# --- Choose an image ID ---
image_path = 'C:/Users/swapn/Downloads/archive/Images/541063517_35044c554a.jpg'
image_id = image_path.split("/")[-1]

if image_id not in image_features:
    raise ValueError(f"Image ID {image_id} not found in image_features.")

# Get its feature vector and generate caption
photo = image_features[image_id].reshape((1, 2048))
caption = generate_caption(photo)

# Output
print("🖼️ Image ID:", image_id)
print("📜 Caption:", caption)

# Show image
display_image_with_caption(image_path, caption)
