INPUT_FILE = "model/light_model.tflite"
OUTPUT_FILE = "model/model.h"

with open(INPUT_FILE, "rb") as f:
    data = f.read()

with open(OUTPUT_FILE, "w") as f:
    f.write("#pragma once\n\n")

    f.write(
        "const unsigned char light_model_tflite[] "
        "__attribute__((aligned(4))) = {\n"
    )

    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write("    ")

        f.write(f"0x{b:02x}")

        if i != len(data) - 1:
            f.write(", ")

        if i % 12 == 11:
            f.write("\n")

    f.write("\n};\n\n")
    f.write(
        f"const unsigned int light_model_tflite_len = {len(data)};\n"
    )

print(f"Created {OUTPUT_FILE}")
print(f"Model size: {len(data)} bytes")