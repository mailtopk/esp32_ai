import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="model/light_model.tflite"
)

interpreter.allocate_tensors()

print(interpreter.get_input_details())
print(interpreter.get_output_details())

# Input:
# shape: [1,6]
# dtype: float32

# Output:
# shape: [1,3]
# dtype: float32