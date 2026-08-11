import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Dense

import pickle


df = pd.read_csv("dataset/generated_dataset.csv")

texts = df["text"].values
labels = df["label"].values


encoder = LabelEncoder()
y = encoder.fit_transform(labels)
tokenizer = Tokenizer(num_words=1000, oov_token="<UNK>")
tokenizer.fit_on_texts(texts)

print("Classes:", encoder.classes_)
print("Number of classes:", len(encoder.classes_))

print("\nClasses found:")

for i, c in enumerate(encoder.classes_):
    print(i, repr(c))

print("\nUnique encoded labels:")
print(np.unique(y))
print("")
print("Rows:", len(df))
print(df.head())
print(df.tail())

MAX_LEN = 8
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(
    X,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

#split test and train data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,  stratify=y
)


np.save(
    "model/X_train.npy",
    X_train
)

VOCAB_SIZE = len(tokenizer.word_index) + 1
print("Vocabulary:", VOCAB_SIZE)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
import tensorflow as tf


# Calculate your exact class count from your data setup
num_classes = len(encoder.classes_) # Should be 3 in your case

model = tf.keras.Sequential([
    # Input matching MAX_LEN = 8 and your 1000 word vocabulary
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16, input_shape=(8,)),
    
    # Flatten the text sequence features
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    
    # CRITICAL: Softmax forces outputs to be 0.0 to 1.0 percentages adding up to 1!
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# CRITICAL: Use categorical classification loss, NOT regression loss
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train and save normally
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
model.save("model/light_model.keras")

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

import json

with open("model/tokenizer.json", "w") as f:
    json.dump(tokenizer.word_index, f)

with open("model/labels.json", "w") as f:
    json.dump(encoder.classes_.tolist(), f)

print("Training complete.")
print("Model Summary")
print(model.inputs)
print(model.input_shape)
print(model.input_dtype)
