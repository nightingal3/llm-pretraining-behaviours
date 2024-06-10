import time

from vllm import LLM, SamplingParams
import argparse

from Mistralv2_prompt import fixed_prompt_mistral
from test_cases import prompts_with_labels


def get_formatted_prompts(texts):
    """
    Adds the Mistral Prompt formatting to an array
    of texts
    """
    formatted_prompts = [
        f"[INST]{fixed_prompt_mistral}{prompt}[/INST]" for prompt in texts
    ]

    return formatted_prompts


def load_model(model_name, temp=0.1, p=0.95):
    """
    Load the model given the model name and sampling params
    """
    sampling_params = SamplingParams(temperature=temp, top_p=p)
    print(f"Initializing LLM with model: {model_name}", flush=True)
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.9,
        swap_space=16,
        tensor_parallel_size=1,
    )
    print("Loaded model!", flush=True)

    return (llm, sampling_params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        type=str,
        required=False,
        help="LLM",
    )
    parser.add_argument(
        "--temperature", default=0.7, type=float, required=False, help="Temperature"
    )
    parser.add_argument("--top_p", default=1, type=float, required=False, help="top_p")
    parser.add_argument("--idx", default=0, type=int, required=False, help="index")

    args = parser.parse_args()
    print("args:", args, flush=True)

    texts = [text[0].replace("\n", "") for text in prompts_with_labels]
    formatted_prompts = get_formatted_prompts(texts)

    sampling_params = SamplingParams(temperature=0.1, top_p=0.95)
    llm_model, sampling_params = load_model(args.model, args.temperature, args.top_p)
    print("Generating outputs...", flush=True)
    outputs = llm_model.generate(formatted_prompts, sampling_params)
    print("Generated outputs...", flush=True)

    output_array = [prompt[1] for prompt in prompts_with_labels]
    true_labels = [True if out == "Code" else False for out in output_array]

    # Print the outputs.
    decisions_array = []
    for output in outputs:
        print("Output- ", output, flush=True)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}", flush=True)
        contains_code = "True" in generated_text
        print("Decision: ", contains_code, flush=True)
        decisions_array.append(contains_code)

    print("True Labels \t Inferenced Label", flush=True)

    correct = 0
    for i in range(len(decisions_array)):
        print(true_labels[i], " \t ", decisions_array[i])
        if true_labels[i] == decisions_array[i]:
            correct += 1

    print("Accuracy = ", correct / len(decisions_array), flush=True)


if __name__ == "__main__":
    main()
