from huggingface_hub import hf_hub_download
from typing import Any
import json
import os
import yaml

DOC_TYPES = [
    "web",
    "web/social_media",
    "web/news",
    "web/blogs",
    "web/forums",
    "books",
    "books/literary/fiction",
    "books/literary/nonfiction",
    "books/textbooks",
    "reference",
    "reference/encyclopedic",
    "reference/dictionaries",
    "academic_papers",
    "academic_papers/sciences",
    "academic_papers/humanities",
    "code",
    "code/source_code",
    "code/documentation",
    "code/forums",
    "media",
    "media/podcasts",
    "media/subtitles",
]
doc_type_prompt = """[STEP 3] What type of dataset is this? Refer to this hierarchy: web/
├─ social_media/
├─ news/
├─ blogs/
├─ forums/
books/
├─ literary/
│  ├─ fiction/
│  ├─ nonfiction/
├─ textbooks/
reference/
├─ encyclopedic/
├─ dictionaries/
academic_papers/
├─ sciences/
├─ humanities/
code/
├─ source_code/
├─ documentation/
├─ forums/
media/
├─ podcasts/
├─ subtitles/
specific_datasets/
├─ <focus of the dataset here, e.g. "finance", "health">/
Enter your choice like this: web/social_media or web or academic_papers/sciences, etc.\nCategory: """


def parse_yaml_metadata(readme_str: str) -> dict:
    """Parse the metadata from the head of a huggingface README.md and return a dictionary"""
    lines = readme_str.split("\n")
    delim = "---"
    delimiter_indices = [i for i, line in enumerate(lines) if line.strip() == delim]

    if len(delimiter_indices) < 2:
        print("Invalid README.md format")
        return {}

    yaml_lines = lines[delimiter_indices[0] + 1 : delimiter_indices[1]]
    yaml_content = "\n".join(yaml_lines)

    try:
        metadata = yaml.safe_load(yaml_content)
        return metadata
    except:
        print("Error parsing metadata from README.md")
        return {}


def get_dataset_metadata_from_hf(
    dataset_name: str, doc_type: str, subdataset: str = ""
) -> dict[str, Any]:
    """Fetch dataset metadata from Hugging Face Hub. Gets numbers for the train split only.
    Note: fetching through load_dataset_builder can take REALLY long, so fetching through README.md
    """
    try:
        readme_path = hf_hub_download(
            repo_id=dataset_name, filename="README.md", repo_type="dataset"
        )
        with open(readme_path, "r") as file:
            readme = file.read()

        yaml_data = parse_yaml_metadata(readme)["dataset_info"]
        if len(yaml_data) > 1 and subdataset != "":
            yaml_data = [d for d in yaml_data if d["name"] == subdataset][0]
        else:
            yaml_data = yaml_data[0]

        if len(yaml_data["splits"]) > 1:
            print(
                "This dataset has multiple splits. We will only fetch metadata for the train split."
            )
            split_data = [s for s in yaml_data["splits"] if s["name"] == "train"][0]
        else:
            split_data = yaml_data["splits"][0]

        metadata = {
            "name": dataset_name,
            "doc_type": doc_type,
            "documents": split_data["num_examples"],
            "data_size_gb": split_data["num_bytes"] / 1e9,
            "tokens_billions": None,  # the hf datasets don't seem to provide this from what I can see
        }
        print("Metadata fetched from huggingface hub.")
        return metadata
    except Exception as e:
        print(f"Error fetching dataset metadata: {e}")
        return None


def enter_dataset_metadata_manually(dataset_name: str, doc_type: str) -> dict[str, Any]:
    """Manually enter dataset metadata."""
    num_documents = int(input("Enter the number of documents in the dataset: "))
    data_size_gb = float(input("Enter the size of the dataset in GB: "))
    tokens_billions = int(
        input(
            "Enter the number of tokens in the dataset in billions (whatever tokenizer used by the paper is fine): "
        )
    )
    metadata = {
        "name": dataset_name,
        "doc_type": doc_type,
        "documents": num_documents,
        "data_size_gb": data_size_gb,
        "tokens_billions": tokens_billions,
    }
    return metadata


def reference_file() -> dict[str, Any]:
    """Enter a file path to reference an existing dataset metadata file."""
    filepath = input(
        "Enter the file path of the dataset metadata you want to include: "
    )
    while not os.path.exists(filepath):
        print("File not found, try again")
        filepath = input("Enter the file path of the dataset metadata: ")

    metadata = {"file_pointer": filepath}
    return metadata


def summarize_and_write_to_file(dataset_metadata: dict, output_filename: str) -> None:
    """Create the dataset summary and write it to a file."""
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    domains = [d for d in dataset_metadata.values()]
    total_tokens_billions = aggregate_tokens_from_metadata(dataset_metadata)
    summary = {"total_size_tokens_billion": total_tokens_billions}

    dataset_metadata_final = {"domains": domains, "summary": summary}
    with open(output_filename, "w") as file:
        json.dump(dataset_metadata_final, file, indent=4)


def aggregate_tokens_from_metadata(metadata: dict) -> int:
    """Aggregate the total number of tokens, including those linked from other files.
    Note: It's important that each file has the summary: {tokens_billions: <number>} field.
    This is checked for in tests and should always be present when using this script to generate metadata.
    """
    total_tokens = 0
    for domain in metadata.values():
        if "file_pointer" in domain:
            with open(domain["file_pointer"], "r") as file:
                data = json.load(file)
                total_tokens += data["summary"]["total_size_tokens_billions"]
        else:
            if domain["tokens_billions"] is not None and isinstance(
                domain["tokens_billions"], (int, float)
            ):
                total_tokens += domain["tokens_billions"]

    return total_tokens


def validate_doc_type(doc_type: str) -> bool:
    """Checks that the user-entered input is in the hierarchy of document types"""
    return doc_type in DOC_TYPES or "specific_dataset" in doc_type


def main():
    # change directory to metadata
    os.chdir(os.path.dirname(__file__))

    datasets_metadata = {}
    print(
        "Recording dataset info now. You can type 'STOP' when asked for a dataset name to finish and save the metadata."
    )
    output_filename = input("Enter the output filename: ")
    if not output_filename.endswith(".json"):
        output_filename += ".json"
    if not output_filename.startswith("datasets_metadata"):
        output_filename = "./dataset_metadata/" + output_filename

    while True:

        entry_option = input(
            "[STEP 1] How do you want to enter the metadata? We can either\n(1) pull from huggingface hub by dataset name, \n(2) reference an existing file under dataset_metadata,\n(3) enter a new dataset manually.\nPlease enter 1, 2, or 3 for these options.\nIf you're not sure, enter 3 and enter the metadata manually. Enter 'stop' to finish.\nYour choice: "
        ).lower()
        if entry_option.lower() == "stop":
            summarize_and_write_to_file(datasets_metadata, output_filename)
            break

        dataset_name = input("[STEP 2] Enter the dataset name or 'STOP' to finish: ")
        if dataset_name.lower() == "stop":
            summarize_and_write_to_file(datasets_metadata, output_filename)
            break

        doc_type = input(doc_type_prompt)

        while not validate_doc_type(doc_type):
            print("Invalid doc type. Please enter a valid doc type.")
            doc_type = input(doc_type_prompt)

        if entry_option == "1":
            subdataset = input(
                "Is there a subset you want to get? For instance, this could be 'en' for the English part. If not, just press enter: "
            )
            metadata = get_dataset_metadata_from_hf(
                dataset_name, doc_type, subdataset=subdataset
            )
            if metadata is None:
                print("Failed to fetch metadata, try entering manually.")
                continue
        elif entry_option == "2":
            metadata = reference_file()
        elif entry_option == "3":
            metadata = enter_dataset_metadata_manually(dataset_name, doc_type)
        else:
            print("Invalid option. Please enter 1, 2, or 3.")
            continue
        print(
            "Here's the metadata for this dataset. Is this right? If not, enter 'abort' to discard the current domain and try again: "
        )
        print(metadata)
        if (
            input(
                "Enter 'abort' to discard the dataset and try again, or just press enter/any other key to proceed: "
            ).lower()
            == "abort"
        ):
            continue
        datasets_metadata[dataset_name] = metadata
        print(
            "A dataset has been recorded. You can now enter another dataset in the data mix or type 'STOP' on the next dataset prompt to finish."
        )

    # Write the gathered metadata to a JSON file
    summarize_and_write_to_file(datasets_metadata, output_filename)

    print(f"Metadata for all entered datasets has been saved to {output_filename}.")


if __name__ == "__main__":
    main()
