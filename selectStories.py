import requests
import os
import random

def getRandom20Stories(content, seed):
    random.seed(seed)
    stories = content.split('- |-')[1:]
    return [stories[i] for i in random.sample(range(len(stories)), 20)]

def fetchStories():
    try:
        url = 'https://huggingface.co/datasets/roneneldan/TinyStories/raw/main/Evaluation%20prompts.yaml'
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print("Error fetching the URL:", response.status_code)
    except requests.RequestException as e:
        print("An error occurred:", e)
    return None

def getGptEval():
    gptEvalDir = "GPT4-eval"
    filePath = "GPT4-eval/20Prompts.txt"
    
    if os.path.exists(filePath):
        return
    os.makedirs(gptEvalDir, exist_ok=True)
    prompts = getRandom20Stories(fetchStories(), 42)
    with open('GPT4-eval/20Prompts.txt', 'w', encoding='utf-8') as file:
                file.write("\n".join(prompts))


if __name__ == '__main__':
    getGptEval()
