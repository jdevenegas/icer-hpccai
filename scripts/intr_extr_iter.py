import requests
import csv
import pandas as pd

# load text from files
with open("system.txt", "r") as f:
    system_prompt = f.read().strip()

with open("example_input.txt", "r") as f:
    example_input = f.read().strip()

with open("example_output.txt", "r") as f:
    example_output = f.read().strip()


def call_llm(system_prompt, example_input, example_output, user_prompt):
    """send prompt to llm and return extracted text"""
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.3:70b",
                "stream": False,
                "options": {
                    "num_ctx": 32768
                },
                "messages": [
                    {"role": "system", "content": system_prompt},

                    {"role": "user", "content": example_input},
                    {"role": "assistant", "content": example_output},

                    {"role": "user", "content": user_prompt}
                ]
            }
        )
        return response.json()['message']['content']
    except Exception as e:
        print(f"Error during Llama call: {e}")
        return None
    
def parse_csv_output(text):
    """parse | delimited CSV text from llm and return list of dicts"""
    lines = text.strip().splitlines()
    if len(lines) <= 1:
        return []  # Only header or empty

    reader = csv.DictReader(lines, delimiter="|")
    return list(reader)

# Iterate over set of excerpts in path_to_text_excerpts file path. 
all_interactions = []
for i in range(218):
    print(f"Processing paragraph {i+1}/{218}...")
    print(f"path-to-text-excerpts/para_{i}.txt")
    with open(f"path-to-text-excerpts/para_{i}.txt", "r") as f:
        user_prompt = f.read().strip()

    result_text = call_llama(system_prompt, example_input, example_output, user_prompt)

    if result_text is None:
        continue  # skip if error

    records = parse_csv_output(result_text)

    if records:
        all_interactions.extend(records)

print(f"Extraction complete. Total interactions found: {len(all_interactions)}")

extr_interactions_df=pd.DataFrame(all_interactions)

extr_interactions_df.to_csv('extracted_interactions.csv')

