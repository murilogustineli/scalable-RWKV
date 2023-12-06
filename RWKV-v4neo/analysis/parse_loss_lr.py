import numpy as np
import re

def parse_file(filepath):
    # Regular expression to match the required pattern
    pattern = r"Epoch 0:.*?loss=(\d+\.\d+).*?lr=(\d+\.\d+)"
    
    # List to store the parsed values
    parsed_values = []

    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("Epoch 0:"):
                # Search for the pattern in the line
                match = re.search(pattern, line)
                if match:
                    # Extract loss and lr values
                    loss, lr = map(float, match.groups())
                    parsed_values.append([loss, lr])

    # Convert the list to a NumPy array
    return np.array(parsed_values)

# Example usage:
# file_path = 'path_to_your_file.txt'
# parsed_array = parse_file(file_path)
# print(parsed_array)
