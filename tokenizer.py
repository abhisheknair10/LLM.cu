import json

with open('model_weights/tokenizer.json') as f:
    d = json.load(f)

print(d)