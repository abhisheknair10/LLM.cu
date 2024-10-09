import json
import ast

# Load the JSON file
with open('model_weights/tokenizer.json', 'r') as f:
    tokenizer = json.load(f)

# Function to recursively replace 'Ġ' with a space in the JSON structure
def replace_char_in_json(obj, old_char, new_char):
    obj = str(obj).replace(old_char, new_char)
    return ast.literal_eval(obj)

# Replace 'Ġ' with space in all strings within the JSON structure
tokenizer = replace_char_in_json(tokenizer, "\u0120", " ")

# Save the modified JSON back to the file
with open('model_weights/modified_tokenizer.json', 'w') as f:
    f.write(tokenizer)

print("Modified JSON has been saved.")