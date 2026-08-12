# Convert keras model to tensorflowlite model

import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers

output_path = "model/light_model.tflite"

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
        return tf.gather(
            self.embedding,
            inputs
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
        })
        return config
    
model = tf.keras.models.load_model(
    "model/light_model.keras",
    custom_objects={
        "EmbeddingLookup": EmbeddingLookup
    }
)

# model = tf.keras.models.load_model(
#     "model/light_model.keras"
# )

# converter = tf.lite.TFLiteConverter.from_keras_model(model)

# tflite_model = converter.convert()

# output_path = "model/light_model.tflite"

# with open(output_path, "wb") as f:
#     f.write(tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open("model/light_model.tflite", "wb") as f:
    f.write(tflite_model)

print(
    "TFLite model:",
    output_path
)

print(
    "Size:",
    os.path.getsize(output_path),
    "bytes"
)