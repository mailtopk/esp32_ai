import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="model/light_model.tflite"
)

interpreter.allocate_tensors()

print("INPUT")

for x in interpreter.get_input_details():
    print("shape:", x["shape"])
    print("dtype:", x["dtype"])

print("\nOUTPUT")

for x in interpreter.get_output_details():
    print("shape:", x["shape"])
    print("dtype:", x["dtype"])

print("\nOPERATORS")

for i, op in enumerate(interpreter._get_ops_details()):
    print(i, op["op_name"])