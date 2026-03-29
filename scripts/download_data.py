"""Download Amazon Reviews 2023 dataset."""

import sys
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    """
    Download file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        desc: Description for progress bar
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
    
    logger.info(f"✅ Downloaded: {output_path}")


def get_amazon_urls(category_id: str) -> dict:
    """
    Get URLs for Amazon dataset files.
    
    Args:
        category_id: Amazon category ID (e.g., 'Electronics')
    
    Returns:
        Dictionary with 'reviews' and 'metadata' URLs
    """
    base_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
    
    return {
        'reviews': f"{base_url}/review_categories/{category_id}.jsonl.gz",
        'metadata': f"{base_url}/meta_categories/meta_{category_id}.jsonl.gz"
    }


def main():
    """Main download function."""
    parser = argparse.ArgumentParser(
        description="Download Amazon Reviews 2023 dataset"
    )
    parser.add_argument(
        '--category',
        type=str,
        required=True,
        help='Category name (e.g., electronics, beauty)'
    )
    parser.add_argument(
        '--skip-reviews',
        action='store_true',
        help='Skip downloading reviews (only download metadata)'
    )
    parser.add_argument(
        '--skip-metadata',
        action='store_true',
        help='Skip downloading metadata (only download reviews)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/raw',
        help='Output directory for downloads'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("📥 Amazon Reviews 2023 - Data Downloader")
    print("="*60 + "\n")
    
    # Load category config
    config_loader = ConfigLoader()
    
    try:
        category_config = config_loader.load_category_config(args.category)
    except FileNotFoundError:
        print(f"❌ Category '{args.category}' not found!")
        print(f"\nAvailable categories: {config_loader.list_available_categories()}")
        return
    
    category_name = category_config['category']['name']
    amazon_id = category_config['category']['amazon_category_id']
    
    print(f"Category: {category_name}")
    print(f"Amazon ID: {amazon_id}\n")
    
    # Get URLs
    urls = get_amazon_urls(amazon_id)
    
    # Setup output directory
    output_dir = Path(args.output_dir) / args.category.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}\n")
    
    # Download reviews
    if not args.skip_reviews:
        reviews_file = output_dir / f"{amazon_id}.jsonl.gz"
        
        if reviews_file.exists():
            response = input(f"⚠️  {reviews_file.name} already exists. Re-download? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("⏭️  Skipping reviews download")
            else:
                print(f"\n📥 Downloading reviews...")
                download_file(urls['reviews'], reviews_file, "Reviews")
        else:
            print(f"\n📥 Downloading reviews...")
            download_file(urls['reviews'], reviews_file, "Reviews")
    
    # Download metadata
    if not args.skip_metadata:
        metadata_file = output_dir / f"meta_{amazon_id}.jsonl.gz"
        
        if metadata_file.exists():
            response = input(f"⚠️  {metadata_file.name} already exists. Re-download? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("⏭️  Skipping metadata download")
            else:
                print(f"\n📥 Downloading metadata...")
                download_file(urls['metadata'], metadata_file, "Metadata")
        else:
            print(f"\n📥 Downloading metadata...")
            download_file(urls['metadata'], metadata_file, "Metadata")
    
    print("\n" + "="*60)
    print("🎉 Download Complete!")
    print("="*60)
    print(f"\nFiles saved to: {output_dir}")
    print("\nNext step:")
    print(f"  python scripts/parse_data.py --category {args.category}")
    print("\n")


if __name__ == "__main__":
    main()
