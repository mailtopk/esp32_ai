import json

with open("model/tokenizer.json") as f:
    vocab=json.load(f)

print("#pragma once")
print()
print("struct WordToken {")
print(" const char* word;")
print(" uint8_t id;")
print("};")
print()
print("WordToken vocabulary[] = {")

for word,idx in vocab.items():
    if word == "<UNK>":
        continue
    print(
        f'{{"{word}",{idx}}},'
    )

print("};")