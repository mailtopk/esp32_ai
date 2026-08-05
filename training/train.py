import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Dense

import pickle


df = pd.read_csv("dataset\commands_dataset.csv")

texts = df["text"].values
labels = df["label"].values

encoder = LabelEncoder()
y = encoder.fit_transform(labels)
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
max_len = 6
X = pad_sequences(
    X,
    maxlen=max_len,
    padding="post"
)

model = Sequential()
model.add(Embedding( input_dim=1000, output_dim=16))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit( X, y,epochs=40, verbose=1)

model.save("model/light_model.keras")

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Training complete.")