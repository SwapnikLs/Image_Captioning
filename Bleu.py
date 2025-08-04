import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

from prepare_sequences import max_length

nltk.download('punkt')

# --- Load model, tokenizer, features, and captions ---
model = load_model('model_caption.keras', compile=False)
with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('data/image_features.pkl', 'rb') as f:
    image_features = pickle.load(f)
with open('data/captions_dict.pkl', 'rb') as f:
    captions_dict = pickle.load(f)

index_word = {v: k for k, v in tokenizer.word_index.items()}

def generate_caption(photo, max_len=max_length):
    in_text = 'startseq'
    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        yhat = model.predict([photo, seq], verbose=0)
        predicted_id = np.argmax(yhat)
        word = index_word.get(predicted_id, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    # Remove startseq/endseq for final result
    return ' '.join(in_text.split()[1:-1])

# --- BLEU Evaluation ---
smooth = SmoothingFunction().method1
bleu_scores = []
sampled_keys = list(captions_dict.keys())[:100]  # Evaluate on 100 images

for img_id in sampled_keys:
    if img_id not in image_features:
        continue
    photo = image_features[img_id].reshape((1, 2048))
    predicted = generate_caption(photo)
    references = [f"startseq {cap.strip()} endseq" for cap in captions_dict[img_id]]
    references = [ref.lower().split()[1:-1] for ref in references]  # Strip startseq/endseq
    candidate = predicted.lower().split()
    bleu = sentence_bleu(references, candidate, weights=(0.5, 0.5), smoothing_function=smooth)
    bleu_scores.append(bleu)

avg_bleu = np.mean(bleu_scores)
print(f"\n🔍 Evaluated on {len(bleu_scores)} samples")
print(f"📈 Average BLEU score (bi-gram): {avg_bleu:.4f}")
