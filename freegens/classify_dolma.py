import json
import os
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from collections import Counter
import pyarrow.parquet as pq
import random
import jsonlines
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Assuming you have set your API key as an environment variable
client = OpenAI()


def detect_language(text: str) -> str:
    try:
        return detect(text[:10000])  # Limit text length for performance
    except LangDetectException:
        return "unknown"


def print_language_distribution(df: pd.DataFrame):
    total = len(df)
    lang_counts = df["language"].value_counts()
    print("\nLanguage Distribution:")
    for lang, count in lang_counts.items():
        percentage = (count / total) * 100
        print(f"{lang}: {count} ({percentage:.1f}%)")


def print_domain_distribution_by_language(df: pd.DataFrame):
    print("\nDomain Distribution by Language:")
    for lang in df["language"].unique():
        lang_df = df[df["language"] == lang]
        total = len(lang_df)
        domain_counts = lang_df["prediction"].value_counts()
        print(f"\n{lang.upper()}:")
        for domain, count in domain_counts.items():
            percentage = (count / total) * 100
            print(f"  {domain}: {count} ({percentage:.1f}%)")


def read_jsonl_sample(
    file_path: str, n_samples: int = 50, text_field: str = "text"
) -> List[str]:
    """Read a random sample of records from a JSONL file."""
    with jsonlines.open(file_path) as reader:
        # Read all records into memory
        records = list(reader)
        records = [r["resps"][0][0] for r in records]

    # Sample records
    if len(records) > n_samples:
        random.seed(43)
        records = random.sample(records, n_samples)

    # Extract text field
    return records


def read_arrow_sample(file_path: str, n_samples: int = 50) -> List[str]:
    """Read a random sample of records from an arrow file efficiently."""
    # Read just the first batch (defaults to 64K rows)
    batch_reader = pq.ParquetFile(file_path).iter_batches(
        batch_size=64 * 1024, columns=["text"]
    )
    first_batch = next(batch_reader)

    # Convert to pandas for easier sampling
    df = first_batch.to_pandas()

    # Sample n_samples random rows
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=43)

    return df["text"].tolist()


def read_samples(
    file_path: str, n_samples: int = 50, text_field: str = "text"
) -> List[str]:
    """Read samples from either Arrow or JSONL files."""
    if file_path.endswith(".arrow"):
        return read_arrow_sample(file_path, n_samples)
    elif file_path.endswith(".jsonl"):
        return read_jsonl_sample(file_path, n_samples, text_field)
    else:
        raise ValueError(f"Unsupported file format for {file_path}")


def extract_top_category(classification: str) -> Tuple[str, str]:
    top_categories = {
        "web": ["social_media", "news", "blogs", "forums", "shopping"],
        "books": ["literary", "fiction", "nonfiction", "textbooks"],
        "reference": ["encyclopedic", "dictionaries"],
        "academic": ["sciences", "humanities"],
        "code": ["source_code", "documentation"],
        "media": ["podcasts", "subtitles"],
        "patent": [],
        "specific_datasets": [],
    }

    # Create a reverse mapping from subcategories to top categories
    subcategory_to_top = {
        sub: top for top, subs in top_categories.items() for sub in subs
    }

    classification_lower = classification.lower()

    for top_category in top_categories.keys():
        if top_category in classification_lower:
            return top_category, classification

    for word in classification_lower.split():
        if word in subcategory_to_top:
            return subcategory_to_top[word], classification

    for subcategory, top_category in subcategory_to_top.items():
        if subcategory in classification_lower:
            return top_category, classification

    return "unknown", classification


def classify_document(document: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": """You are a system tasked with classifying documents into different categories based on the knowledge they contain and their source. Some documents may be messy or not fit cleanly into one category, but try to find the best fit based on the taxonomy below.\n\n.
├── web/
│   ├── social_media
│   ├── news
│   ├── blogs
│   ├── forums
│   └── shopping
├── books/
│   ├── literary_fiction
│   └── literary_nonfiction
├── reference/
│   ├── encyclopedic
│   ├── dictionaires
│   └── textbooks
├── academic/
│   ├── sciences
│   ├── humanities
│   └── patents
├── code/
│   ├── source_code
│   ├── documentation
│   └── forums
├── media/
│   ├── podcasts
│   └── subtitles
└── specific_datasets (health, finance, etc)
If a text is incoherent, please return the category that it seems to be trying to emulate. However, if this is completely impossible, return "incoherent".

If a text has hyperlinks, messages to "click here" or similar, user handles or email addresses, it's probably a "web" document. However, if it has significant code that is not HTML or CSS (e.g. stackoverflow posts), it should be classified as "code".
""",
                    }
                    #             {
                    #                 "role": "system",
                    #                 "content": [
                    #                     {
                    #                         "type": "text",
                    #                         "text": """You are a system tasked with classifying documents into different categories based on the knowledge they contain and their source. Some documents may be messy or not fit cleanly into one category, but try to find the best fit based on the taxonomy below.\n\n.
                    # ├── web/
                    # │   ├── social_media
                    # │   ├── news
                    # │   ├── blogs
                    # │   ├── forums
                    # │   └── shopping
                    # ├── books/
                    # │   ├── literary_fiction
                    # │   └── literary_nonfiction
                    # ├── reference/
                    # │   ├── encyclopedic
                    # │   ├── dictionaires
                    # │   └── textbooks
                    # ├── academic/
                    # │   ├── sciences
                    # │   ├── humanities
                    # │   └── patents
                    # ├── code/
                    # │   ├── source_code
                    # │   ├── documentation
                    # │   └── forums
                    # ├── media/
                    # │   ├── podcasts
                    # │   └── subtitles
                    # └── specific_datasets (health, finance, etc)
                    # A note on commonly confused examples: materials aimed primarily at teaching or explaining concepts (like textbooks or encyclopedias) should be classified as "reference" rather than literary nonfiction. Literary nonfiction should be reserved for works that are primarily narrative or argumentative in nature.
                    # Documentation about code or config files (markdown, yaml, etc) should be classified as 'code' rather than 'reference'. If the document is a patent, classify it as 'patent'. If the document is a specific dataset, classify it as 'specific_datasets'. If you are unsure, respond with 'unknown'.
                    # The reference domain primarily consists of wikipedia articles and how-to guides. If you see formatting similar to wikipedia (headers, references, and years in brackets), it is likely a wikipedia article (rather than literary nonfiction). If the first phrase is the title of the article, followed by a few sentences of summary, it is likely a wikipedia article. If the document is a list of definitions or a step by step explanation of how to do something, it should also fall under this category.
                    # Examples:
                    # Web: Bauer Vapor X600 junior ice hockey skates S17 now available in store and online!
                    # The Bauer Vapor X600 Skates debut the new Fiber Composite boot construction that is stiffer and lighter than year's past. This translates into more power and better longevity since it better resists breaking down over time. The new one-piece injected core helps too because it helps to bolster the internal structure.
                    # books: CHAPTER III
                    # KIARTAN AT CRAGNESS
                    # On the morning of the fifth day thereafter, as Rolf stood by the gate
                    # of the enclosure which protected the farm buildings, he saw a man
                    # coming on a horse, and knew him for his father's brother Kiartan. He
                    # was a big man, heavily bearded, dressed in bright-colored clothes and
                    # hung about with gold chains. His eye was bright and roving; his face
                    # was genial, and he looked about him as he came as one who is well
                    # contented. Yet Rolf liked him not.
                    # [...]
                    # reference: Karl, Prince of Leiningen (1804–1856)
                    # Karl, Prince of Leiningen, KG (Karl Friedrich Wilhelm Emich; 12 September 1804 – 13 November 1856) was the third Prince of Leiningen and maternal half-brother of Queen Victoria. Leiningen served as a Bavarian lieutenant general, before he briefly played an important role in German politics as the first Prime Minister of the ""Provisorische Zentralgewalt"" government formed by the Frankfurt Parliament in 1848.
                    # [...]
                    # """,}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f'Document:{document}\nCategory (one of "web", "books", "reference", "academic", "code", "media", "patent" or a specific dataset not represented here. Only respond with the words above and not the specific category). If a text is incoherent, please return the category that it seems to be trying to emulate. However, if this is completely impossible, return "incoherent".\nCategory:',
                    }
                ],
            },
        ],
        temperature=0,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "text"},
    )
    return response.choices[0].message.content.strip()


def evaluate_accuracy(predictions: List[str], ground_truth: List[str]) -> Dict:
    """Calculate accuracy metrics for the classifications."""
    correct = sum(
        1 for p, g in zip(predictions, ground_truth) if p.lower() == g.lower()
    )
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    # Create confusion matrix
    confusion = Counter([(g, p) for g, p in zip(ground_truth, predictions)])

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "confusion_matrix": confusion,
    }


def print_confusion_matrix(confusion: Counter, total_samples: int):
    """Print a nicely formatted confusion matrix with percentages."""
    # Get unique labels maintaining order
    unique_labels = sorted(set([label for pair in confusion.keys() for label in pair]))

    # Calculate width needed for labels
    max_label_length = max(len(label) for label in unique_labels)
    cell_width = max(max_label_length + 2, 10)  # Minimum 10 chars wide

    # Print header
    print("\nConfusion Matrix (counts and percentages of total):")
    print("-" * (cell_width + len(unique_labels) * cell_width + 3))

    # Header row
    header = " " * cell_width + "│ "
    header += " ".join(label.rjust(cell_width - 1) for label in unique_labels)
    print(header)
    print("-" * (cell_width) + "┼" + "-" * (len(unique_labels) * cell_width + 1))

    # Print each row
    for true_label in unique_labels:
        row = [true_label.ljust(cell_width - 1) + "│"]
        for pred_label in unique_labels:
            count = confusion.get((true_label, pred_label), 0)
            percentage = (count / total_samples) * 100
            cell = f"{count:>3d} ({percentage:4.1f}%)"
            row.append(cell.rjust(cell_width))
        print(" ".join(row))

    print("-" * (cell_width + len(unique_labels) * cell_width + 3))


def main():
    base_path = (
        "/data/tir/projects/tir5/users/mengyan3/dolma_data_processed/dolma_100B_deduped"
    )
    samples_per_domain = 500

    # Define the domains and their ground truth labels
    domain_files = {
        "web": [
            (f"{base_path}/c4/part_1.arrow", "web"),
            (f"{base_path}/common-crawl/part_1.arrow", "web"),
        ],
        "books": [(f"{base_path}/gutenberg-books/part_1.arrow", "books")],
        "academic": [(f"{base_path}/peS2o/part_1.arrow", "academic")],
        "code": [(f"{base_path}/stack-code/part_1.arrow", "code")],
        "reference": [(f"{base_path}/wiki-en-simple/part_1.arrow", "reference")],
        # "nl_0_code_100": [
        #     ("/data/tir/projects/tir6/general/mengyan3/tower-llm-training/freegens/generations/220m_nl_0_code_100.jsonl", None)
        # ],
        # "nl_80_code_20": [
        #     ("/data/tir/projects/tir6/general/mengyan3/tower-llm-training/freegens/generations/220m_nl_80_code_20.jsonl", None)
        # ],
        # "nl_100_code_0": [
        #     ("/data/tir/projects/tir6/general/mengyan3/tower-llm-training/freegens/generations/220m_nl_100_code_0.jsonl", None)
        # ]
        # "phi-2": [
        #     ('/data/tir/projects/tir6/general/mengyan3/tower-llm-training/freegens/generations/phi2.jsonl', None)
        # ],
        # "qwen-1.5": [
        #     ('/data/tir/projects/tir6/general/mengyan3/tower-llm-training/freegens/generations/qwen1.5_batch1.jsonl', None)
        # ],
        # "qwen-2.5": [
        #     ('/data/tir/projects/tir6/general/mengyan3/tower-llm-training/freegens/generations/qwen2.5_batch1.jsonl', None)
        # ],
        # "olmo-7b": [
        #     ('/data/tir/projects/tir6/general/mengyan3/tower-llm-training/freegens/generations/olmo500.jsonl', None)
        # ]
        # "phi-1.5": [
        #     ('/data/tir/projects/tir6/general/mengyan3/tower-llm-training/freegens/generations/phi1.5_batch1.jsonl', None)
        # ]
    }

    results = []
    all_predictions = []
    all_ground_truth = []

    for domain, files in domain_files.items():
        print(f"\nProcessing {domain} domain...")

        for file_path, ground_truth_label in files:
            try:
                # Read samples from the arrow file
                samples = read_samples(file_path, samples_per_domain)

                # Classify each sample
                for text in tqdm(samples, desc=f"Classifying {domain} samples"):
                    if len(text) > 128000 * 3.5:
                        # approx # chars to tokens in en is 4
                        text = text[: int(128000 * 3.5)]
                    language = detect_language(text)
                    prediction = classify_document(text)
                    top_category, _ = extract_top_category(prediction)
                    results.append(
                        {
                            "domain": domain,
                            "text": text,
                            "prediction_raw": prediction,
                            "prediction": top_category,
                            "language": language,
                        }
                    )

                    if ground_truth_label:
                        results[-1]["ground_truth"] = ground_truth_label
                        all_ground_truth.append(ground_truth_label)
                    all_predictions.append(top_category)

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

    # Calculate overall accuracy

    if not all(domain_type is None for domain_type in all_ground_truth):
        accuracy_metrics = evaluate_accuracy(all_predictions, all_ground_truth)

        # Save results
        df = pd.DataFrame(results)
        df.to_csv("./freegens/domain_classification_results_dolma.csv", index=False)

        # Print results
        print("\nClassification Results:")
        print(f"Overall Accuracy: {accuracy_metrics['accuracy']:.2%}")
        print(
            f"Correct Classifications: {accuracy_metrics['correct']} out of {accuracy_metrics['total']}"
        )

        # Print confusion matrix
        print_confusion_matrix(
            accuracy_metrics["confusion_matrix"], accuracy_metrics["total"]
        )

        # Calculate per-domain accuracy
        print("\nPer-Domain Accuracy:")
        for domain in domain_files.keys():
            domain_preds = [
                p for r, p in zip(results, all_predictions) if r["domain"] == domain
            ]
            domain_truth = [r["ground_truth"] for r in results if r["domain"] == domain]
            domain_accuracy = evaluate_accuracy(domain_preds, domain_truth)
            print(f"{domain}: {domain_accuracy['accuracy']:.2%}")
    else:
        # just print overall predictions
        print("\nClassification Results:")
        print(f"Predictions:")
        df = pd.DataFrame(results)
        print(df["prediction"].value_counts() / df.shape[0])
        print("Excluding unknowns:")
        print(
            df[df["prediction"] != "unknown"]["prediction"].value_counts()
            / df[df["prediction"] != "unknown"].shape[0]
        )

        # print lang distribution
        print_language_distribution(df)
        df.to_csv("./freegens/domain_classification_test_129.csv", index=False)


if __name__ == "__main__":
    main()
