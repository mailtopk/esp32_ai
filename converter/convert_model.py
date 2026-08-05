import tensorflow as tf


model = tf.keras.models.load_model("model/light_model.keras")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with open("model/light_model.tflite","wb") as f:
    f.write(tflite_model)

print("done")