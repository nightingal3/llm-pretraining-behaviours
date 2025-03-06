import json


def load_json_documents(file_path, start_index, end_index):
    text_lengths = []
    with open(file_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i >= start_index and i < end_index:
                try:
                    document = json.loads(line)
                    text_length = len(
                        document["text"]
                    )  # Assuming 'text' is the key for the text content
                    text_lengths.append((i, text_length, document["text"]))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {i}: {e}")
                except KeyError as e:
                    print(f"Missing expected key {e} on line {i}")
                except Exception as e:
                    print(f"An error occurred on line {i}: {e}")
            elif i >= end_index:
                break
    return text_lengths


def print_largest_documents(text_lengths, number_of_documents=5):
    # Sort the documents by text length in descending order and print the largest ones
    sorted_lengths = sorted(text_lengths, key=lambda x: x[1], reverse=True)
    for index, length, text in sorted_lengths[:number_of_documents]:
        print(f"Document at line {index}: text length = {length}")
        print(
            f"Text snippet: {text[:20]}..."
        )  # Prints the first 200 characters of the text


def count_large_documents(file_path, length_threshold):
    large_document_count = 0
    total_documents = 0
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                document = json.loads(line)
                text_length = len(
                    document["text"]
                )  # Assuming 'text' is the key for the text content
                if text_length > length_threshold:
                    large_document_count += 1
                total_documents += 1
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
            except KeyError as e:
                print(f"Missing expected key {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

    return large_document_count, total_documents


def filter_and_save_json(file_path, output_path, length_threshold):
    large_document_indices = []
    total_documents = 0
    kept_documents = 0

    with open(file_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        for i, line in enumerate(infile):
            try:
                document = json.loads(line)
                text_length = len(
                    document["text"]
                )  # Assuming 'text' is the key for the text content
                if text_length > length_threshold:
                    large_document_indices.append(i)
                else:
                    json.dump(document, outfile)
                    outfile.write("\n")
                    kept_documents += 1
                total_documents += 1
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {i}: {e}")
            except KeyError as e:
                print(f"Missing expected key {e} on line {i}")
            except Exception as e:
                print(f"An error occurred on line {i}: {e}")

    return large_document_indices, total_documents, kept_documents


# Define the path to your JSON file and the indices of documents you want to inspect
file_path = "/data/tir/projects/tir3/users/mengyan3/dolma_1T_remaining_domains/gutenberg-books-deduped.json"
output_path = "/data/tir/projects/tir3/users/mengyan3/dolma_1T_remaining_domains/gutenberg-books-deduped-cutoff.json"

start_index = 1000  # Adjust to the index of the first document you want to inspect
end_index = (
    2000  # Adjust to one past the index of the last document you want to inspect
)
length_threshold = 1000000  # Define your length threshold here

# Load the documents and their text lengths
text_lengths = load_json_documents(file_path, 0, float("inf"))

# Print the largest documents
print_largest_documents(text_lengths)

large_count, total_count = count_large_documents(file_path, length_threshold)
print(
    f"Number of documents with text length greater than {length_threshold}: {large_count} out of {total_count} total documents"
)
large_docs, total_docs, kept_docs = filter_and_save_json(
    file_path, output_path, length_threshold
)
