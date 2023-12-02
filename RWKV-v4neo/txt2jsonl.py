import json

# Replace 'input.txt' with your text file's name
with open('data/TinyStories-valid.txt', 'r') as txt_file:
    # Replace 'output.jsonl' with your desired output file's name
    with open('data/TinyStories-valid.jsonl', 'w') as jsonl_file:
        for line in txt_file:
            # Process the line into a JSON object
            # This example assumes each line is a simple text entry
            json_object = {"text": line.strip()}

            # Write the JSON object to the .jsonl file
            jsonl_file.write(json.dumps(json_object) + '\n')
