import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("../model/light_model.keras")
with open("../model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
with open("model/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

while True:
    sentence = input("> ")
    seq = tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences( seq, maxlen=6, padding="post")
    prediction = model.predict(seq, verbose=0)
    label = np.argmax(prediction)

    print(
        encoder.inverse_transform([label])[0],
        prediction
    )