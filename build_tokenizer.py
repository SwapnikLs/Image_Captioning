import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Load captions_dict
with open('data/captions_dict.pkl', 'rb') as f:
    captions_dict = pickle.load(f)

# Collect all captions with startseq and endseq tokens
all_captions = []
for captions in captions_dict.values():
    for cap in captions:
        cap = f"startseq {cap.strip()} endseq"
        all_captions.append(cap)

# Fit tokenizer (with <unk> as the OOV token)
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)

# Save tokenizer
with open('data/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"Tokenizer fitted on {len(tokenizer.word_index)} unique words.")
# --- Tokenization Check Snippet ---

# Pick a sample caption
sample_caption = "startseq a dog runs through the grass endseq"

# Convert the words to their corresponding token ids
tokens = tokenizer.texts_to_sequences([sample_caption])[0]
print(f"Caption: {sample_caption}")
print(f"Token IDs: {tokens}")

# Build index-to-word mapping
index_word = {v: k for k, v in tokenizer.word_index.items()}

# Convert token ids back to words
reconstructed = [index_word.get(tok, "<unk>") for tok in tokens]
print(f"Reconstructed: {' '.join(reconstructed)}")
