#!/usr/bin/env python3

import os
import sys
import argparse
import zipfile
from pathlib import Path
from tqdm import tqdm


def extract_zip_files(source_dir, dest_dir, verbose=True):
    """
    Extract all .zip files from source_dir to dest_dir.
    
    Args:
        source_dir (str): Source directory containing .zip files
        dest_dir (str): Destination directory for extracted files
        verbose (bool): Print progress information
    
    Returns:
        tuple: (num_extracted, num_skipped, num_failed)
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Validate source directory
    if not source_path.exists():
        print(f"❌ Error: Source directory does not exist: {source_dir}")
        return 0, 0, 0
    
    if not source_path.is_dir():
        print(f"❌ Error: Source path is not a directory: {source_dir}")
        return 0, 0, 0
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"📂 Source directory: {source_path.absolute()}")
        print(f"📂 Destination directory: {dest_path.absolute()}")
    
    # Find all .zip files
    zip_files = list(source_path.glob("*.zip"))
    
    if not zip_files:
        print(f"⚠️  No .zip files found in {source_dir}")
        return 0, 0, 0
    
    if verbose:
        print(f"🔍 Found {len(zip_files)} .zip file(s) to extract\n")
    
    num_extracted = 0
    num_failed = 0
    num_skipped = 0
    
    for zip_file in zip_files:
        try:
            if verbose:
                print(f"📦 Extracting: {zip_file.name}")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                members = zip_ref.infolist()

                for member in tqdm(members, desc=f"Extracting {zip_file.name}", unit="file"):
                    zip_ref.extract(member, dest_path)
            
            if verbose:
                print(f"✅ Successfully extracted: {zip_file.name}")
            num_extracted += 1
            
        except zipfile.BadZipFile:
            print(f"❌ Error: {zip_file.name} is not a valid zip file")
            num_failed += 1
        except Exception as e:
            print(f"❌ Error extracting {zip_file.name}: {str(e)}")
            num_failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 Extraction Summary:")
    print(f"   ✅ Successfully extracted: {num_extracted}")
    print(f"   ❌ Failed: {num_failed}")
    print(f"   ⏭️  Skipped: {num_skipped}")
    print(f"{'='*60}\n")
    
    return num_extracted, num_skipped, num_failed


def main():
    parser = argparse.ArgumentParser(
        description="Extract .zip dataset files from source to destination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from ./data to ./data_extracted
  python extract_dataset.py
  
  # Extract from custom locations
  python extract_dataset.py --source /path/to/zips --dest /path/to/output
  
  # Quiet mode (minimal output)
  python extract_dataset.py --source /path/to/zips --dest /path/to/output --quiet
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='../../../../../mnt/nfs-share/AI_Datasets/V2Xverse',
        help='Source directory containing .zip files (default: ./data)'
    )
    parser.add_argument(
        '--dest',
        type=str,
        default='../../../../../mnt/nfs-share/AI_Datasets/_unzipped/V2Xverse',
        help='Destination directory for extracted files (default: ./data_extracted)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output (only show errors and summary)'
    )
    
    args = parser.parse_args()
    
    num_extracted, num_skipped, num_failed = extract_zip_files(
        args.source,
        args.dest,
        verbose=not args.quiet
    )
    
    # Exit with appropriate code
    sys.exit(0 if num_failed == 0 else 1)


if __name__ == '__main__':
    main()
