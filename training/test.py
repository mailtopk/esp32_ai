import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("model/light_model.keras")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("*"*80)
print("Model Summary")
print("*"*80)
print(model.summary())
print("*"*80)
print()


with open("model/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

print("*"*80)
print(f"Tokenizer len {len(tokenizer.word_index)}")
print(f"Encoder : {encoder.classes_}")
print(f"Tokenizer : {tokenizer.word_index}")
print("*"*80)

while True:
    sentence = input("> ")
    seq = tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences( seq, maxlen=6, padding="post")
    prediction = model.predict(seq, verbose=0)
    label = np.argmax(prediction)
    confidence = max(prediction[0])
    command = encoder.inverse_transform([prediction.argmax()])[0]

    if confidence < 0.80:
        command = "UNKNOWN"

    print(command, confidence)
    print(
        encoder.inverse_transform([label])[0],
        prediction
    )