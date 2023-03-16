from pathlib import Path
import numpy as np
import json
'''
This script creates a random list of word lengths for us to evaluate the model on in terms of length controllability.

The data is stored in the folder wordlength_eval_data.

The data is a json file with the following format:
{
    "text": "<len> num_words <text>", "num_words": num_words
}
'''

if __name__ == "__main__":
    num_examples = 50
    max_words = 30

    for seed in range(10):
        np.random.seed(seed)
        # This randomly generates the number of words we want
        num_words_ls = np.random.randint(low=2, high=max_words+1, size=num_examples)
        with open(f'wordlength_eval_data/wordlength_eval_data_{seed}.json', 'w') as f:
            for i in range(num_examples):
                num_words = num_words_ls[i]

                prefix = f"<len> {num_words} <text>"
                line = json.dumps({'text': prefix, 'num_words': str(num_words)})
                if i < num_examples:
                    line += '\n'
                f.write(line)
