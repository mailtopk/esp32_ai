import tensorflow as tf
import numpy as np

MODEL = "model/light_model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]


def run(ids):
    x = np.array([ids], dtype=np.int32)

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()

    return interpreter.get_tensor(output_index)[0]


tests = {
    "ESP32 turn on": [
        28, 2, 5, 15, 0, 0, 0, 0
    ],

    "ESP32 turn off": [
        28, 2, 5, 13, 0, 0, 0, 0
    ],

    "ESP32 please turn bedroom on": [
        6, 28, 2, 22, 5, 15, 0, 0
    ],

    "ESP32 hello": [
        45, 0, 0, 0, 0, 0, 0, 0
    ],

    "ESP32 make room brighter": [
        20, 2, 17, 1, 0, 0, 0, 0
    ],
}


for name, ids in tests.items():

    output = run(ids)

    print()
    print(name)
    print("Input :", ids)
    print("Output:", output.tolist())
    print("Class :", int(np.argmax(output)))