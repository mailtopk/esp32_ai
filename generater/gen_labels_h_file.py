import json

INPUT_FILE = "model/labels.json"
OUTPUT_FILE = "model/labels.h"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    labels = json.load(f)

def escape_cpp_string(value):
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("#pragma once\n\n")
    f.write("static const char* const kLabels[] = {\n")
    for label in labels:
        escaped = escape_cpp_string(label)
        f.write(f'    "{escaped}",\n')

    f.write("};\n\n")
    f.write(
        "static const int kNumLabels = "
        "(sizeof(kLabels) / sizeof(kLabels[0]));\n"
    )

print("===================================")
print("Labels header generated")
print("===================================")
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Number of labels: {len(labels)}")
print()

for i, label in enumerate(labels):
    print(f"{i}: {label}")