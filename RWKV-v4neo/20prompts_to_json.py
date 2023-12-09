import json


def convert_text2json(file_path: str="20Prompts.txt", save: bool=True) -> str:
    # Read the file and split it into prompts
    with open(file_path, 'r') as file:
        prompts = file.read().split('\n\n')

    # Remove any empty strings in case there are extra new lines
    prompts = [f"{i+1}: {prompt[3:]}" for i, prompt in enumerate(prompts) if prompt]

    # Convert the list of prompts into JSON format
    json_data = json.dumps(prompts, indent=4)

    if save:
        # Save the JSON data to a file
        with open('20prompts.json', 'w') as json_file:
            json_file.write(json_data)
    
    return json_data

def main(verbose=False):
    json_file = convert_text2json()
    if verbose:
        print(json_file)

if __name__ == '__main__':
    main()
