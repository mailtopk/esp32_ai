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


X = tokenizer.texts_to_sequences(texts)
max_len = 6
X = pad_sequences(
    X,
    maxlen=max_len,
    padding="post"
)

#split test and train data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,  stratify=y
)


model = Sequential()
model.add(Embedding( input_dim=1000, output_dim=32))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation="relu"))
model.add(Dense(len(encoder.classes_), activation="softmax"))

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

import json

with open("model/tokenizer.json", "w") as f:
    json.dump(tokenizer.word_index, f)

with open("model/labels.json", "w") as f:
    json.dump(encoder.classes_.tolist(), f)

print("Training complete.")
