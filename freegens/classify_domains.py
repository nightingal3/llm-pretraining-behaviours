import json
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from openai import OpenAI
import os

client = OpenAI()


def detect_language(text: str) -> str:
    """Detect the language of the text."""
    try:
        return detect(text[:10000])  # Limit text length for performance
    except LangDetectException:
        return "unknown"


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
        "incoherent": [],
        "unknown": [],
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


def read_samples(file_path: str, text_field: str = "text") -> List[str]:
    """Read all text samples from a file (supports JSONL format)."""
    if file_path.endswith(".jsonl"):
        import jsonlines

        with jsonlines.open(file_path) as reader:
            if text_field == "resps":
                return [record["resps"][0][0] for record in reader]
            return [record[text_field] for record in reader]
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def process_file(
    input_file: str, output_file: str, resp_format: bool = False, limit: int = None
):
    """Process the input file to generate domain distribution and save results."""
    print(f"Reading data from {input_file}...")

    # Read samples
    try:
        samples = read_samples(
            input_file, text_field="resps" if resp_format else "text"
        )
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    results = []
    if limit:
        samples = samples[:limit]

    print("Classifying documents...")
    for text in tqdm(samples, desc="Processing documents"):
        if len(text) < 50:  # skip if text is too short
            results.append(
                {
                    "text": text,
                    "language": "unknown",
                    "prediction_raw": "too_short",
                    "prediction": "incoherent",
                }
            )
        language = detect_language(text)
        prediction = classify_document(text)
        top_category = extract_top_category(prediction)
        results.append(
            {
                "text": text,
                "language": language,
                "prediction_raw": prediction,
                "prediction": top_category[0],
            }
        )

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Calculate and display domain distribution
    print("\nDomain Distribution:")
    domain_counts = df["prediction"].value_counts()
    total = len(df)
    for domain, count in domain_counts.items():
        percentage = (count / total) * 100
        print(f"{domain}: {count} ({percentage:.1f}%)")

    # Save results to output file
    print(f"Saving results to {output_file}...")
    # mkdir -p
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate domain distribution for a file."
    )
    parser.add_argument(
        "--input", type=str, help="Path to the input file (JSONL format)."
    )
    parser.add_argument(
        "--output", type=str, help="Path to the output file (CSV format)."
    )
    parser.add_argument(
        "--resps_format", action="store_true", help="Use 'resps' field for input data."
    )
    parser.add_argument(
        "--limit", type=int, help="Limit the number of samples to process."
    )

    args = parser.parse_args()
    process_file(args.input, args.output, args.resps_format, args.limit)
