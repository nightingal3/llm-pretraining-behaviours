import argparse
import os

json_escape_sequences = [r'"', r"\", r"/", r"\b", r"\f", r"\n", r"\t"]

def clean_file(input: str, output: str) -> None:
    # if no output is passed, input file will be overwritten
    if output is None:
        output = input
    with open(input, 'r') as file:
        data = file.read()
    for sequence in json_escape_sequences:
        escaped_sequence = f"\{sequence}"
        data = data.replace(sequence, escaped_sequence)

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=False)
    args = parser.parse_args()
    
    input = args.input
    output = args.output
    clean_file(input, output)

if __name__ == "__main__":
    