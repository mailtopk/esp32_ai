import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

test_sentences = [
    "turn the light on",
    "turn the light off",
    "please turn the bedroom light on",
    "hello",
    "make the room brighter"
]

tokenizer = Tokenizer(num_words=1000, oov_token="<UNK>")
# Build the vocabulary
tokenizer.fit_on_texts(test_sentences)

sequences = tokenizer.texts_to_sequences(test_sentences)
print("Number of sequences:", len(sequences))

for i, seq in enumerate(sequences):
    print(i, repr(seq), [type(x) for x in seq])

padded = tf.keras.preprocessing.sequence.pad_sequences(
    sequences,  maxlen=8, padding="post", truncating="post")

for text, seq in zip(test_sentences, padded):
    print(repr(text))
    print(seq.tolist())
    print()