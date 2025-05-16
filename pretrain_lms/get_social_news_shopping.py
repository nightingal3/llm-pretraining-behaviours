import json
import os
import tqdm
import argparse
import hashlib
from datasets import load_dataset
import random

def categorize_by_url(url):
    """
    Categorize a URL as 'web' if it contains any web-related terms,
    rather than using strict domain checking.
    """
    if not url:
        return "unknown"
        
    url = url.lower()
    
    # Web-related terms for social, news, blogs, and shopping
    web_terms = [
        # Social terms
        "reddit", "twitter", "facebook", "instagram", "tumblr", "pinterest", 
        "linkedin", "quora", "tiktok", "snapchat", "forum", "community", 
        "social", "discuss", "thread", "chat", "comment",
        
        # News terms
        "news", "cnn", "bbc", "nytimes", "reuters", "ap", "washingtonpost", 
        "theguardian", "wsj", "forbes", "cnbc", "aljazeera", "npr", "foxnews", 
        "bloomberg", "times", "post", "herald", "tribune", "journal", "gazette",
        
        # Blog terms
        "medium", "wordpress", "blogger", "blogspot", "substack", "blog", 
        "opinion", "editorial", "column",
        
        # Shopping terms
        "amazon", "ebay", "walmart", "etsy", "shopify", "aliexpress", "shop", 
        "store", "product", "buy", "market", "commerce", "retail", "cart", 
        "shopping", "deal", "price", "discount", "offer", "sale", "coupon"
    ]
    
    # Check if URL contains any web terms
    if any(term in url for term in web_terms):
        return "web"
    else:
        return "non_web"

def extract_web_content(output_path, max_rows=100000000, batch_size=1000, refinedweb_name="tiiuae/falcon-refinedweb"):
    """
    Extract web content from RefinedWeb dataset using streaming.
    
    Args:
        output_path: Path to save the extracted web content
        max_rows: Maximum number of rows to extract
        batch_size: Batch size for processing
        refinedweb_name: Name of the RefinedWeb dataset on HuggingFace
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load dataset with streaming
    try:
        dataset = load_dataset(refinedweb_name, split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Try alternative loading method
        try:
            print("Trying alternative loading method...")
            dataset = load_dataset(refinedweb_name, streaming=True)["train"]
        except Exception as e2:
            print(f"Failed to load dataset: {e2}")
            return
    
    # Shuffle the dataset to get a good mix
    dataset = dataset.shuffle(buffer_size=10000, seed=42)
    
    # Extract web content
    extracted_count = 0
    processed_count = 0
    
    # Create two progress bars
    dataset_pbar = tqdm.tqdm(desc="Processed documents", unit=" docs")
    web_pbar = tqdm.tqdm(desc="Extracted web content", total=max_rows, unit=" docs")
    
    with open(output_path, 'w') as f:
        for doc in dataset:
            # Update processed count
            processed_count += 1
            dataset_pbar.update(1)
            
            # Extract URL and text
            url = doc.get('url', '')
            text = doc.get('content', '')
            
            # Skip if no text or URL
            if not text or not url:
                continue
            
            # Categorize based on URL
            category = categorize_by_url(url)
            # Keep only web content
            if category == "web":
                # Generate a stable ID based on text content
                doc_id = hashlib.md5(text.encode('utf-8')).hexdigest()
                
                # Create simplified document
                simple_doc = {
                    "id": doc_id,
                    "text": text
                }
                
                # Write document
                f.write(json.dumps(simple_doc) + '\n')
                extracted_count += 1
                web_pbar.update(1)
                
                # Show occasional stats
                #if extracted_count % 10000 == 0:
                    #web_ratio = extracted_count / processed_count
                    #dataset_pbar.set_postfix(web_ratio=f"{web_ratio:.1%}")
                
                # Check if we've reached the limit
                if extracted_count >= max_rows:
                    break
    
    # Close progress bars
    dataset_pbar.close()
    web_pbar.close()
    
    print(f"Processed {processed_count} documents total")
    print(f"Extracted {extracted_count} web documents to {output_path}")
    print(f"Web content ratio: {extracted_count/processed_count:.1%}")

def create_web_mixtures(web_content_path, output_dir, web_percentages=[30, 50, 70, 90], docs_per_mixture=10000):
    """
    Create different mixtures with varying web content percentages.
    
    Args:
        web_content_path: Path to the extracted web content
        output_dir: Directory to save mixtures
        web_percentages: List of web percentages to create
        docs_per_mixture: Number of documents per mixture
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating web content mixtures...")
    
    # Read the web content
    web_docs = []
    with open(web_content_path, 'r') as f:
        for line in tqdm.tqdm(f, desc="Reading web content"):
            web_docs.append(json.loads(line))
    
    print(f"Read {len(web_docs)} web documents")
    
    # Create dummy non-web content (could be replaced with actual non-web content)
    # Here we're just using placeholder text
    non_web_docs = []
    for i in range(max(docs_per_mixture, 100000)):  # Generate enough dummy docs
        non_web_docs.append({
            "id": f"nonweb_{i}",
            "text": f"This is a non-web document with ID {i}. It contains technical, educational, or reference material that would not be classified as social media, news, blogs, or shopping content."
        })
    
    # Create mixtures
    for web_pct in web_percentages:
        output_path = os.path.join(output_dir, f"web_{web_pct}_mixture.jsonl")
        
        # Calculate document counts
        web_count = int(docs_per_mixture * (web_pct / 100))
        non_web_count = docs_per_mixture - web_count
        
        print(f"Creating mixture with {web_pct}% web content:")
        print(f"  - Web documents: {web_count}")
        print(f"  - Non-web documents: {non_web_count}")
        
        # Sample documents
        sampled_web = random.sample(web_docs, min(web_count, len(web_docs)))
        sampled_non_web = random.sample(non_web_docs, min(non_web_count, len(non_web_docs)))
        
        # Combine and shuffle
        mixture = sampled_web + sampled_non_web
        random.shuffle(mixture)
        
        # Write mixture to file
        with open(output_path, 'w') as f:
            for doc in mixture:
                f.write(json.dumps(doc) + '\n')
        
        print(f"Created mixture with {web_pct}% web content: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract web content from RefinedWeb")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_rows", type=int, default=100000000, help="Maximum number of rows to extract")
    parser.add_argument("--dataset", default="tiiuae/falcon-refinedweb", help="RefinedWeb dataset name")
    parser.add_argument("--create_mixtures", action="store_true", help="Create web content mixtures")
    parser.add_argument("--percentages", nargs="+", type=int, default=[30, 50, 70, 90], 
                        help="Web percentages for mixtures")
    
    args = parser.parse_args()
    
    # Extract web content
    web_content_path = os.path.join(args.output_dir, "web_content.jsonl")
    extract_web_content(
        output_path=web_content_path,
        max_rows=args.max_rows,
        refinedweb_name=args.dataset
    )
    
    # Optionally create mixtures
    if args.create_mixtures:
        create_web_mixtures(
            web_content_path=web_content_path,
            output_dir=args.output_dir,
            web_percentages=args.percentages
        )