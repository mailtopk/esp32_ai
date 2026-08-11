import tensorflow as tf
import numpy as np
import json

MODEL = "model/light_model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

print("Input:", input_details[0]["shape"], input_details[0]["dtype"])
print("Output:", output_details[0]["shape"], output_details[0]["dtype"])

test_inputs = [
    ("turn the light on",       [3, 2, 4, 5, 0, 0, 0, 0]),
    ("turn the light off",      [3, 2, 4, 6, 0, 0, 0, 0]),
    ("please turn the bedroom light on",
                                [7, 3, 2, 8, 4, 5, 0, 0]),
    ("hello",                   [9, 0, 0, 0, 0, 0, 0, 0]),
    ("make the room brighter",  [10, 2, 11, 12, 0, 0, 0, 0]),
]

for text, ids in test_inputs:

    x = np.array([ids], dtype=np.int32)

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()

    output = interpreter.get_tensor(output_index)[0]

    predicted_class = int(np.argmax(output))

    print()
    print("Text:", text)
    print("Input:", ids)
    print("Output:", output.tolist())
    print("Predicted class:", predicted_class)

import json

with open("model/labels.json", "r") as f:
    labels = json.load(f)

print("Labels:")
for i, label in enumerate(labels):
    print(i, repr(label))