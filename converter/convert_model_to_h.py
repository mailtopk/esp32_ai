# alternate xxd -i light_model.tflite > model.h

input_file = "model/light_model.tflite"
output_file = "esp32/mcesp32s3/src/model.h"

with open(input_file, "rb") as f:
    data = f.read()

with open(output_file, "w") as f:
    f.write( "const unsigned char light_model_tflite[] = {\n")
    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write("\n    ")

        f.write(f"0x{b:02x}, ")

    f.write("\n};\n\n" )
    f.write(f"const unsigned int light_model_tflite_len = {len(data)};\n" )

print(f"Created {output_file}, size={len(data)} bytes")