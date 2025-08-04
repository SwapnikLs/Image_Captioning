# build_captions_dict.py

import pandas as pd
import pickle
from collections import defaultdict

# Load captions.txt (with header)
df = pd.read_csv('C:/Users/swapn/Downloads/archive/captions.txt')

# Group captions by image filename
captions_dict = defaultdict(list)
for _, row in df.iterrows():
    img = row['image']
    cap = row['caption']
    captions_dict[img].append(cap)

# Save dictionary to pickle file
with open('data/captions_dict.pkl', 'wb') as f:
    pickle.dump(dict(captions_dict), f)

print(f"Saved captions_dict with {len(captions_dict)} images.")
