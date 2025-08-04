import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.callbacks import ModelCheckpoint

# -------- Load Training Data -------- #
X1 = np.load('data/X1.npy')              # Image features
X2 = np.load('data/X2.npy')              # Input sequences
y = np.load('data/y.npy')                # Next word indices

# -------- Load Tokenizer -------- #
with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_length = X2.shape[1]

# -------- Model Architecture -------- #
# Image feature input
inputs1 = Input(shape=(X1.shape[1],))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# Caption sequence input
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# Decoder (merge both inputs)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# Final model and compilation
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

model.summary()

# -------- Training -------- #
checkpoint = ModelCheckpoint('model_caption.h5', monitor='loss', save_best_only=True, verbose=1)
model.fit([X1, X2], y, epochs=20, batch_size=256, callbacks=[checkpoint])

# Optional: Save final Keras model
model.save('model_caption.keras')
