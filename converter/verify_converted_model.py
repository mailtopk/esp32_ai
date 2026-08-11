import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="model/light_model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("INPUT DETAILS")
for x in input_details:
    print(x)

print("\nOUTPUT DETAILS")
for x in output_details:
    print(x)