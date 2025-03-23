"""
Preprocessing script for X-ray classification project.
Run this script to preprocess the DICOM files and create the training dataset.

Example usage:
    python preprocess.py --augment-level medium --val-split 0.2 --img-size 256
"""

import argparse
import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import XRayPreprocessor

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess DICOM X-ray images for classification.')
    
    parser.add_argument('--base-path', type= str, default = './',
                        help='Base path containing data folder')
    
    parser.add_argument('--output-path', type =str, default = './processed_data',
                        help='Path to save processed images')
    
    parser.add_argument('--img-size', type= int, default=256,
                        help='Target image size (both width and height)')
    
    parser.add_argument(' --augment', action = 'store_true', default = True,
                        help='Apply data augmentation')
    
    parser.add_argument('--augment-level', type=str, default = 'medium', 
                        choices =['none', 'light', 'medium', 'heavy'],
                        help ='Level of augmentation to apply')
    
    parser.add_argument('--val-split', type= float, default = 0.2,
                        help='Proportion of data to use for validation')
    
    parser.add_argument('--analyze', action ='store_true', default=True,
                        help= 'Analyze the dataset after processing')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create preprocessor
    processor = XRayPreprocessor(
        base_path = args.base_path ,
        output_path = args.output_path,
        img_size = (args.img_size, args.img_size)
    )
    
    print("Loading metadata...")
    processor.load_metadata()
    
    print(f"Processing images with {args.augment_level} augmentation level...")
    processor.process_images(
        augment=args.augment,
        val_split=args.val_split,
        augment_level=args.augment_level
    )
    
    if args.analyze:
        print("Analyzing dataset..." )
        stats = processor.analyze_dataset()
        print("\nDataset Statistics:" )
        print(stats)
    
    print("\nPreprocessing complete!")
    print(f"Processed images saved to: {args.output_path}")

if __name__ == "__main__" :
    main()