import json
import os
import pandas as pd
import argparse

def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    model_name = data['model_name']
    results = {}

    if 'results' in data and 'harness' in data['results']:
        harness_data = data['results']['harness']
        for task, task_data in harness_data.items():
            # Consolidate hendrycksTest and mmlu
            if task.startswith('hendrycksTest-'):
                task = f"mmlu-{task.split('-', 1)[1]}"
            elif task.startswith('mmlu-'):
                pass  # Keep mmlu_ prefix as is
            
            if isinstance(task_data, dict):
                if any(key.endswith('-shot') for key in task_data.keys()):
                    # Format with n-shot nesting
                    for shot, shot_data in task_data.items():
                        for metric, value in shot_data.items():
                            if isinstance(value, (int, float)):
                                if task.startswith('mmlu-'):
                                    # For mmlu tasks, remove shot information
                                    results[f"{task}_{metric}"] = value
                                else:
                                    results[f"{task}_{shot}_{metric}"] = value
                else:
                    # Format without n-shot nesting
                    for metric, value in task_data.items():
                        if isinstance(value, (int, float)):
                            results[f"{task}_{metric}"] = value

    return model_name, results

def main(input_dir, output_file):
    all_results = []

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            model_name, results = process_json_file(file_path)
            results['model_name'] = model_name
            all_results.append(results)

    df = pd.DataFrame(all_results)
    df.set_index('model_name', inplace=True)
    df.to_csv(output_file)
    print(f"CSV file has been created: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON results to CSV")
    parser.add_argument("input_dir", help="Directory containing JSON files")
    parser.add_argument("output_file", help="Output CSV file name")
    args = parser.parse_args()

    main(args.input_dir, args.output_file)