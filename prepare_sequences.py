import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load required files ---
with open('data/captions_dict.pkl', 'rb') as f:
    captions_dict = pickle.load(f)

with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('data/image_features.pkl', 'rb') as f:
    image_features = pickle.load(f)

# --- Calculate max_length using all captions with start/end tokens ---
all_captions = []
for captions in captions_dict.values():
    for cap in captions:
        all_captions.append(f"startseq {cap.strip()} endseq")
max_length = max(len(c.split()) for c in all_captions)
vocab_size = len(tokenizer.word_index) + 1

X1, X2, y = [], [], []

# --- Create training sequences ---
for img_id, captions in captions_dict.items():
    if img_id not in image_features:
        continue
    feature = image_features[img_id]
    for cap in captions:
        # Ensure every caption is wrapped in startseq and endseq
        cap_full = f"startseq {cap.strip()} endseq"
        cap_seq = tokenizer.texts_to_sequences([cap_full])[0]
        # For each position in the sequence, create a (image, partial caption) -> next word target
        for i in range(1, len(cap_seq)):
            in_seq = cap_seq[:i]
            out_seq = cap_seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)

# --- Convert lists to numpy arrays ---
X1 = np.array(X1)
X2 = np.array(X2)
y = np.array(y, dtype='int32')

# --- Save arrays for training ---
np.save('data/X1.npy', X1)
np.save('data/X2.npy', X2)
np.save('data/y.npy', y)

print(f"Saved sequences: X1 shape {X1.shape}, X2 shape {X2.shape}, y shape {y.shape}, max_length: {max_length}")
