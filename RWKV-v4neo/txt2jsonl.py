import json

def split_stories(text):
    # Splitting stories based on the specific separator line
    separator = '<|endoftext|>'
    return text.split(separator)

def convert_to_jsonl(stories, output_file):
    with open(output_file, 'w') as file:
        for story in stories:
            # Cleaning and trimming each story
            story_clean = story.strip().replace('\n', ' ')
            if story_clean:  # Ensure the story is not empty
                # Creating a JSON object for each story
                json_object = json.dumps({"text": story_clean})
                file.write(json_object + '\n')

def main():
    input_file = './data/TinyStories-train.txt'  # Replace with your input file path
    output_file = './data/TinyStories-train.jsonl'  # Replace with your output file path

    with open(input_file, 'r') as file:
        text = file.read()

        stories = split_stories(text)
        convert_to_jsonl(stories, output_file)

    print(f"Converted stories to {output_file}")

if __name__ == "__main__":
    main()
