import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="model/light_model.tflite"
)

for op in interpreter._get_ops_details():
    print(op["op_name"])