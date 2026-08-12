import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="model/light_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

# example:
# "switch on light"
x = np.array( [[12,5,8,3,0,0]], dtype=np.float32)

interpreter.set_tensor( input_details[0]["index"], x)
interpreter.invoke()

result = interpreter.get_tensor(output_details[0]["index"])

print(result)