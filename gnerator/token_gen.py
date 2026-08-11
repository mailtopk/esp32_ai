import json

INPUT_FILE = "model/tokenizer.json"
OUTPUT_FILE = "model/tokenizer_vocab.h"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# JSON stores IDs as numbers, but make sure everything is normalized.
items = sorted(
    ((word, int(token_id)) for word, token_id in vocab.items()),
    key=lambda x: x[1]
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("#pragma once\n\n")
    f.write("#include <stdint.h>\n\n")

    f.write("struct VocabEntry {\n")
    f.write("    const char* word;\n")
    f.write("    int32_t id;\n")
    f.write("};\n\n")

    f.write("static const VocabEntry kVocabulary[] = {\n")

    for word, token_id in items:
        # Escape C/C++ string characters
        escaped = (
            word
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )

        f.write(
            f'    {{"{escaped}", {token_id}}},\n'
        )

    f.write("};\n\n")

    f.write(
        "static const size_t kVocabularySize = "
        "sizeof(kVocabulary) / sizeof(kVocabulary[0]);\n"
    )

print("Created:", OUTPUT_FILE)
print("Vocabulary entries:", len(items))

for word in (
    "turn",
    "the",
    "light",
    "on",
    "off",
    "please",
    "bedroom",
    "hello",
    "make",
    "room",
    "brighter",
):

    print(word, vocab.get(word))