import csv
import random
import pandas as pd
from pyparsing import Combine

random.seed(42)

# -----------------------
# LIGHT ON
# -----------------------
import pandas as pd
import random
templates = {
    "LIGHT_ON": [
        "turn on the {}",
        "switch on the {}",
        "activate the {}",
        "{} on",
        "power on the {}",
        "make it bright",
        "make the room bright",
        "brighten the room",
        "light up the room",
        "it is dark",
        "its dark",
        "too dark in here",
        "I cannot see",
        "need some light",
        "wake up the lamp",
    ],

    "LIGHT_OFF": [
        "turn off the {}",
        "switch off the {}",
        "disable the {}",
        "{} off",
        "power off the {}",
        "make it dark",
        "darken the room",
        "lights are too bright",
        "too much light",
        "I want darkness",
        "go to sleep lamp",
    ],

    "UNKNOWN": [
        "hello",
        "hi",
        "tell me a joke",
        "what is the weather",
        "what time is it",
        "play music",
        "open the door",
    ]
}

objects = [
    "light",
    "lamp",
    "LED",
    "bedroom light",
    "kitchen light"
]


prefixes = [
    "",
    "",
    "please",
    "can you",
    "could you",
    "hey",
    "hey assistant"
]


def add_noise(text):
    prefix = random.choice(prefixes)
    if prefix:
        return prefix + " " + text
    return text


def generate_dataset(samples_per_class=500):
    data = []

    for label, template_list in templates.items():
        for _ in range(samples_per_class):
            template = random.choice(template_list)
            # Create command
            if "{}" in template:
                obj = random.choice(objects)
                text = template.format(obj)
            else:
                text = template

            # Add noise HERE
            text = add_noise(text)

            data.append({
                "text": text,
                "label": label
            })

    return pd.DataFrame(data)


df = generate_dataset(500)

print(df.sample(20))
print(df["label"].value_counts())
    
df_r = pd.read_csv("commands_dataset.csv")
combined = pd.concat([df, df_r], ignore_index=True)

combined.to_csv("generated_dataset.csv")

print(combined.head())
print(combined["label"].value_counts())
print("Dataset saved.")

