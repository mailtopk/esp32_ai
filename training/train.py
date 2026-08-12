import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

from tensorflow import keras
import pickle

# mbedding layer maintains a learned table roughly like "GATHER"
class EmbeddingLookup(layers.Layer):
    def __init__(self, vocab_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.embedding = self.add_weight(
            name="embedding",
            shape=(self.vocab_size, self.embedding_dim),
            initializer="uniform",
            trainable=True,
        )

    def call(self, inputs):
        return tf.gather(self.embedding, inputs)

    def get_config(self):
        config = super().get_config()
        config.update({ "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim})
        return config


# Read training data
df = pd.read_csv("dataset/generated_dataset.csv")

texts = df["text"].values
labels = df["label"].values # 3 labels


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

MAX_LEN = 8 # padding
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(   X,  maxlen=MAX_LEN, padding="post",  truncating="post")

#split test and train data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,  stratify=y
)



VOCAB_SIZE = len(tokenizer.word_index) + 1
print("Vocabulary:", VOCAB_SIZE)

# 1000 × 16 = 16,000 parameters
VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
NUM_CLASSES = len(encoder.classes_)

MAX_LEN = 8
NUM_CLASSES = len(encoder.classes_)

model = tf.keras.Sequential([
    tf.keras.Input( shape=(MAX_LEN,), dtype=tf.int32, name="token_ids" ),
    EmbeddingLookup(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        name="embedding_lookup"
    ),
    layers.GlobalAveragePooling1D(),
    layers.Dense( 16, activation="relu" ),
    layers.Dense(  NUM_CLASSES, activation="softmax" )
])

model.compile( optimizer="adam", loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
model.fit( X_train, y_train,  epochs=30, validation_data=(X_test, y_test))
model.save("model/light_model.keras")
model = tf.keras.models.load_model(
    "model/light_model.keras", custom_objects={"EmbeddingLookup": EmbeddingLookup })

print(model.summary)


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