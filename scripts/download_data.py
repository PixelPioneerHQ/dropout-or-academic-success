#!/usr/bin/env python
"""
Script to download the 'Dropout or Academic Success' dataset from Kaggle.
You need to have a Kaggle account and API token set up.
See: https://github.com/Kaggle/kaggle-api#api-credentials
"""

import os
import zipfile
import kaggle
import argparse

def download_dataset(output_dir='./data'):
    """
    Download the dataset from Kaggle and extract it to the specified directory.
    
    Args:
        output_dir (str): Directory to save the dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset identifier on Kaggle
    dataset_name = "ankanhore545/dropout-or-academic-success"
    
    print(f"Downloading dataset '{dataset_name}' to {output_dir}...")
    
    try:
        # Download the dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=True
        )
        print(f"Dataset successfully downloaded and extracted to {output_dir}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nMake sure you have set up your Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click on 'Create New API Token'")
        print("3. Save the kaggle.json file to ~/.kaggle/kaggle.json")
        print("4. Run 'chmod 600 ~/.kaggle/kaggle.json' to set permissions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Kaggle dataset')
    parser.add_argument('--output', type=str, default='../data',
                        help='Directory to save the dataset')
    
    args = parser.parse_args()
    download_dataset(args.output)