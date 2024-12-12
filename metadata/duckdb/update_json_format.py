import json
from pathlib import Path
from collections import defaultdict

def get_default_shot_setting(benchmark):
    """Get default shot setting for each benchmark"""
    defaults = {
        'arc': '25-shot',
        'hellaswag': '10-shot',
        'hendrycksTest': '5-shot',  # all mmlu tests
        'truthfulqa': '0-shot',
        'winogrande': '5-shot',
        'lambada': '0-shot',
        'drop': '3-shot',
        'gsm8k': '5-shot',
        'arithmetic': '5-shot',
        'minerva': '5-shot',
        'mathqa': '5-shot',
        'xnli': '0-shot',
        'anli': '0-shot',
        'logiqa2': '0-shot',
        'fld': '0-shot',
        'asdiv': '5-shot'
    }
    
    # Find matching prefix
    for prefix, setting in defaults.items():
        if benchmark.startswith(prefix):
            return setting
    return '0-shot'  # default fallback

def normalize_metric_name(old_name):
    """Normalize metric names to consistent format"""
    metric_map = {
        'exact_match': 'acc',
        'em': 'acc',
        'acc': 'acc',
        'accuracy': 'acc'
    }
    return metric_map.get(old_name, old_name)

def update_json_format(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nProcessing files...")
    for json_path in input_dir.glob("*.json"):
        try:
            print(f"\nProcessing {json_path}")
            with open(json_path) as f:
                data = json.load(f)
            
            results = data.get("results", {})
            new_results = {}
            
            # Handle harness results
            if "harness" in results:
                harness_results = results.pop("harness")
                new_harness = {}
                
                for benchmark, content in harness_results.items():
                    if isinstance(content, dict):
                        # Handle case where there's already a shot setting
                        if any('shot' in key for key in content.keys()):
                            if 'x-shot' in content:
                                # Replace x-shot with proper default
                                shot_setting = get_default_shot_setting(benchmark)
                                metrics = content['x-shot']
                                new_metrics = {}
                                for k, v in metrics.items():
                                    metric_name = normalize_metric_name(k)
                                    stderr_name = f"{metric_name}_stderr"
                                    if 'stderr' in k.lower():
                                        continue
                                    new_metrics[metric_name] = v
                                    stderr_val = metrics.get(f"{k}_stderr", metrics.get(f"{k}_std", metrics.get("std", None)))
                                    if stderr_val and stderr_val != "N/A":
                                        new_metrics[stderr_name] = stderr_val
                                new_metrics['timestamp'] = metrics.get('timestamp')
                                new_harness[benchmark] = {shot_setting: new_metrics}
                            else:
                                # Just normalize the metric names
                                new_content = {}
                                for shot_setting, metrics in content.items():
                                    new_metrics = {}
                                    for k, v in metrics.items():
                                        metric_name = normalize_metric_name(k)
                                        stderr_name = f"{metric_name}_stderr"
                                        if 'stderr' in k.lower():
                                            continue
                                        new_metrics[metric_name] = v
                                        stderr_val = metrics.get(f"{k}_stderr", metrics.get(f"{k}_std", metrics.get("std", None)))
                                        if stderr_val and stderr_val != "N/A":
                                            new_metrics[stderr_name] = stderr_val
                                    new_metrics['timestamp'] = metrics.get('timestamp')
                                    new_content[shot_setting] = new_metrics
                                new_harness[benchmark] = new_content
                        else:
                            # Need to add shot setting
                            shot_setting = get_default_shot_setting(benchmark)
                            new_metrics = {}
                            for k, v in content.items():
                                metric_name = normalize_metric_name(k)
                                stderr_name = f"{metric_name}_stderr"
                                if 'stderr' in k.lower():
                                    continue
                                new_metrics[metric_name] = v
                                stderr_val = content.get(f"{k}_stderr", content.get(f"{k}_std", content.get("std", None)))
                                if stderr_val and stderr_val != "N/A":
                                    new_metrics[stderr_name] = stderr_val
                            new_metrics['timestamp'] = content.get('timestamp')
                            new_harness[benchmark] = {shot_setting: new_metrics}
                            
                if new_harness:
                    new_results["harness"] = new_harness
            
            # Handle non-harness results
            for benchmark, content in results.items():
                if isinstance(content, dict):
                    if 'x-shot' in content:
                        # Replace x-shot with proper default
                        shot_setting = get_default_shot_setting(benchmark)
                        new_results[benchmark] = {shot_setting: content['x-shot']}
                    elif any('shot' in key for key in content.keys()):
                        new_results[benchmark] = content
                    else:
                        shot_setting = get_default_shot_setting(benchmark)
                        new_results[benchmark] = {shot_setting: content}
            
            data["results"] = new_results
            
            # Write to new directory
            output_path = output_dir / json_path.name
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            print(f"Written to {output_path}")
            
        except Exception as e:
            print(f"Error processing {json_path}: {str(e)}")
            raise  # Re-raise to see full traceback

if __name__ == "__main__":
    input_dir = "/data/tir/projects/tir6/general/mengyan3/tower-llm-training/metadata/model_scores"
    output_dir = "/data/tir/projects/tir6/general/mengyan3/tower-llm-training/metadata/model_scores_new"
    update_json_format(input_dir, output_dir)