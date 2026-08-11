import re
import tensorflow as tf

HEADER_FILE = "model/model.h"

with open(HEADER_FILE, "r", encoding="utf-8") as f:
    header = f.read()

hex_values = re.findall(r"0x([0-9a-fA-F]{2})", header)

model_bytes = bytes(int(x, 16) for x in hex_values)

print("Model bytes:", len(model_bytes))

interpreter = tf.lite.Interpreter(
    model_content=model_bytes
)

interpreter.allocate_tensors()

print("\nINPUT:")
for x in interpreter.get_input_details():
    print("shape:", x["shape"])
    print("dtype:", x["dtype"])

print("\nOUTPUT:")
for x in interpreter.get_output_details():
    print("shape:", x["shape"])
    print("dtype:", x["dtype"])

print("\nOPERATORS:")

for i, op in enumerate(interpreter._get_ops_details()):
    print(i, op["op_name"])