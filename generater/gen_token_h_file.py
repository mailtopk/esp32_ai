# Generate Tokenizer for ESP32 device

import json
INPUT_FILE = "model/tokenizer.json"
OUTPUT_FILE = "model/tokenizer_vocab.h"
# Must match your Keras tokenizer / Embedding.
MAX_VOCAB_SIZE = 1000

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    word_index = json.load(f)

entries = []
for word, token_id in word_index.items():
    token_id = int(token_id)
    # Token IDs >= 1000 cannot be used by Embedding(input_dim=1000)
    if token_id < MAX_VOCAB_SIZE:
        entries.append((word, token_id))

# Sort by token ID, which makes the generated file easy to inspect.
entries.sort(key=lambda x: x[1])

def escape_cpp_string(value):
    return (
        value
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("#pragma once\n\n")
    f.write("#include <stdint.h>\n\n")

    f.write("struct VocabEntry {\n")
    f.write("    const char* word;\n")
    f.write("    int32_t id;\n")
    f.write("};\n\n")
    f.write("static const VocabEntry kVocabulary[] = {\n")

    for word, token_id in entries:
        escaped_word = escape_cpp_string(word)
        f.write(
            f'    {{"{escaped_word}", {token_id}}},\n'
        )
    f.write("};\n\n")
    f.write(
        "static const int kVocabularySize = "
        "(sizeof(kVocabulary) / sizeof(kVocabulary[0]));\n"
    )


print()
print("===================================")
print("Tokenizer header generated")
print("===================================")
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Entries: {len(entries)}")
print()

print("First vocabulary entries:")

for word, token_id in entries[:20]:
    print(f"{token_id:4d} -> {word}")