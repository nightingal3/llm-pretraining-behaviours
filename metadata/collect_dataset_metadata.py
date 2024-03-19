from datasets import load_dataset_builder
from typing import Any
import json
import os


def get_dataset_metadata_from_hf(dataset_name: str, doc_type: str) -> dict[str, Any]:
    """Fetch dataset metadata from Hugging Face Hub. Gets numbers for the train split only."""
    try:
        builder = load_dataset_builder(dataset_name)
        breakpoint()
        metadata = {
            "name": dataset_name,
            "doc_type": doc_type,
            "documents": builder.info.splits["train"]["num_examples"],
            "data_size_gb": builder.info.splits["train"]["num_bytes"] / 1e9,
            "tokens_billions": None,
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
    tokens_billions = float(
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
    filepath = input("Enter the file path of the dataset metadata: ")
    while not os.path.exists(filepath):
        print("File not found, try again")
        filepath = input("Enter the file path of the dataset metadata: ")

    metadata = {"file_pointer": filepath}
    return metadata


def summarize_and_write_to_file(dataset_metadata: dict, output_filename: str) -> None:
    """Create the dataset summary and write it to a file."""
    raise NotImplementedError


def main():
    datasets_metadata = {}
    print(
        "Recording dataset info now. You can type 'STOP' when asked for a dataset name to finish and save the metadata."
    )
    output_filename = input("Enter the output filename: ")
    if not output_filename.endswith(".json"):
        output_filename += ".json"
    if not output_filename.startswith("datasets_metadata"):
        output_filename = "dataset_metadata/" + output_filename

    while True:

        entry_option = input(
            "How do you want to enter the metadata? We can either (1) pull from huggingface hub by dataset name, (2) reference an existing file under dataset_metadata, (3) or enter a new dataset manually. Please enter 1, 2, or 3 for these options. "
        ).lower()

        dataset_name = input("Enter the dataset name or 'STOP' to finish: ")
        if dataset_name.lower() == "stop":
            summarize_and_write_to_file(datasets_metadata, output_filename)
            break

        doc_type = input(
            """What type of dataset is this? Refer to this hierarchy: web/
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
Enter your choice like this: web/social_media or web or academic_papers/sciences, etc.: """
        )

        if entry_option == "1":
            metadata = get_dataset_metadata_from_hf(dataset_name, doc_type)
            if metadata is None:
                print("Failed to fetch metadata, try entering manually.")
                continue
        elif entry_option == "2":
            raise NotImplementedError
        elif entry_option == "3":
            metadata = enter_dataset_metadata_manually()
        else:
            print("Invalid option. Please enter 1, 2, or 3.")
            continue
        print(
            "Here's the metadata for this dataset. Is this right? If not, enter 'abort' to discard the dataset and try again: "
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
    with open("datasets_metadata.json", "w") as file:
        json.dump(datasets_metadata, file, indent=4)

    print(f"Metadata for all entered datasets has been saved to {output_filename}.")


if __name__ == "__main__":
    main()
