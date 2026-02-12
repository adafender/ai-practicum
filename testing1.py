import json

with open('personas.json', 'r') as file:
    data = json.load(file)
    for key, value in data.items():
        for sub_key, sub_value in value.items():
            print(f"{sub_key}: {sub_value}")