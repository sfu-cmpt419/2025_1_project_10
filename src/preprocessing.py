"""
Preprocessing module for X-ray image classification project.
This module handles:
- DICOM to JPG conversion
- Image normalization
- Data augmentation (GaussianBlur and HorizontalFlip)
- Dataset preparation
"""

import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random
from tqdm import tqdm
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

class XRayPreprocessor:
    """
    Class for preprocessing X-ray DICOM images for machine learning.
    """
    
    def __init__(self,  base_path="./", output_path="./processed_data", img_size=( 256, 256 )):
        """
        Initialize the preprocessor.
        
        Args:
            base_path: Base directory containing the data
            output_path: Directory to save processed images
            img_size: Target image size as width height
        """
        self.base_path = base_path
        self.output_path = output_path
        self.img_size = img_size
        self.labels = [

            "Abdomen", "Ankle", "Cervical Spine", "Chest", "Clavicles",
            "Elbow", "Feet", "Finger", "Forearm", "Hand", "Hip", "Knee",
            "Lower Leg", "Lumbar Spine", "Others", "Pelvis", "Shoulder",
            "Sinus", "Skull", "Thigh", "Thoracic Spine", "Wrist"
        ]
        self.train_df = None
        
    def _create_directories(self):
        """Create directories for processed images""" 
        
        os.makedirs(self.output_path, exist_ok = True)
        os.makedirs(os.path.join(self.output_path, "train"), exist_ok = True )
        os.makedirs(os.path.join( self.output_path, "val"), exist_ok = True)
    
        for label in self.labels:
            os.makedirs(os.path.join(self.output_path, "train", label ), exist_ok = True )
            os.makedirs(os.path.join(self.output_path, "val", label), exist_ok = True)
    
    def load_metadata( self, metadata_path = "data/train.csv" ):
        """
        Load and prepare metadata from CSV
        
        Args:
            metadata_path: Path to the CSV file with metadata
            
        Returns:
            DataFrame with metadata
        """
        file_path = os.path.join( self.base_path, metadata_path )
        df = pd.read_csv(file_path)
        
        # Add label names
        df["Name" ] = [ self.labels[int(x.split(" ")[0])] for x in df["Target" ]]
        self.train_df = df
        return df
    
    def find_dicom_file(self, uid, base_dir = "data/train/"):
        """
        Find a DICOM file path based on its UID
        
        Args:
            uid: The SOPInstanceUID
            base_dir: Base directory to search in
            
        Returns:
            Path to the DICOM file or None if not found
        """
        search_pattern = f"{uid }-c.dcm"
        for root, dirs, files in os.walk( os.path.join ( self.base_path, base_dir)):
            if search_pattern in files:
                return os.path.join(root, search_pattern )
        return None
    
    def _normalize_dicom(self, dicom_data ):
        """
        Normalize DICOM pixel array
        
        Args:
            dicom_data: The DICOM dataset
            
        Returns:
            Normalized pixel array as a PIL Image
        """
        # getitng pixel array
        pixel_array = dicom_data.pixel_array
        
        #standard z-score normalization
        mean = np.mean(pixel_array )
        std = np.std( pixel_array) + 1e-8 
        normalized = (pixel_array - mean ) / std
        
        #scale to [0, 1]
        normalized = ( normalized + 3) / 6
        normalized = np.clip(normalized, 0, 1)
        
        # Convert to 8bit
        img_array = (normalized * 255).astype(np.uint8 )
        
        #create PIL image
        img = Image.fromarray( img_array )
        return img
    
    def _resize_with_padding(self, img):
        """
        Resize image to target size with padding
        
        Args:
            img: PIL Image
            
        Returns:
            Resized and padded image
        """
        # Calculate aspect ratio
        width, height = img.size
        aspect = width / height    
        
        # Determine new size preserving aspect ratio
        if aspect > 1:
            new_width = self.img_size[0]    
            new_height = int(new_width / aspect )
        else:
            new_height = self.img_size[1]
            new_width = int(new_height * aspect)
        
        # Resize image
        img = img.resize(( new_width, new_height ), Image.LANCZOS)    
        
        # Create padded image
        padded_img = Image.new('L', self.img_size, 0)
        
        # Paste resized image onto padded image
        x_offset = (self.img_size[0] - new_width) // 2    
        y_offset = (self.img_size[1] - new_height) // 2
        padded_img.paste(img, (x_offset, y_offset)) 
        
        return padded_img
    
    def _apply_gaussian_blur(self, img, radius_range = (0, 1.5)):   
        """
        Apply random Gaussian blur to image
        
        Args:
            img: PIL Image
            radius_range: Range for blur radius
            
        Returns:
            Blurred image
        """
        radius = random.uniform(*radius_range)
        if radius > 0.1:
            return img.filter(ImageFilter.GaussianBlur( radius = radius ))    
        return img
    
    def _apply_horizontal_flip( self, img, probability=0.5):
        """
        Randomly flip image horizontally
        
        Args:
            img: PIL Image
            probability: Probability of applying the flip
            
        Returns:
            Possibly flipped image
        """
        if random.random() < probability:
            return img.transpose( Image.FLIP_LEFT_RIGHT)     
        return img
    
    def _apply_rotation(self,img, angle_range=(-15, 15)):
        """
        Apply random rotation to image
        
        Args:
            img: PIL Image
            angle_range: Range for rotation angle in degrees
            
        Returns:
            Rotated image
        """
        angle = random.uniform(*angle_range )
        if abs(angle) > 0.5:
            return img.rotate(angle, resample= Image.BILINEAR, expand=False, fillcolor=0)
        return img
    
    def _apply_brightness_adjustment(self, img, factor_range = (0.85, 1.15)):
        """
        Apply random brightness adjustment
        
        Args:
            img: PIL Image
            factor_range: Range for brightness adjustment factor
            
        Returns:
            Brightness-adjusted image
        """
        factor = random.uniform( *factor_range )
        enhancer = ImageEnhance.Brightness(img )
        return enhancer.enhance( factor)
    
    def _apply_contrast_adjustment(self, img, factor_range = (0.9, 1.1 )):
        """
        Apply random contrast adjustment
        
        Args:
            img: PIL Image
            factor_range: Range for contrast adjustment factor
            
        Returns:
            Contrast-adjusted image
        """
        factor = random.uniform(*factor_rang )
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def _apply_random_noise( self, img, intensity_range = (0, 10)):
        """
        Apply random noise to the image
        
        Args:
            img: PIL Image
            intensity_range: Range for noise intensity
            
        Returns:
            Noisy image
        """
        intensity = random.uniform(*intensity_range)
        if intensity < 0.1: 
            return img
            
        #convert to numpy array
        img_array = np.array(img).astype(np.float32)
        
        # Add Gaussian noise
        noise = np.random.normal(0, intensity, img_array.shape)
        img_array = img_array + noise
        
        # Clip values to valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        return Image.fromarray(img_array)
    
    def _apply_augmentations(self, img, augment_level='none'):
        """
        Apply a series of augmentations to the image
        
        Args:
            img: PIL Image
            augment_level: Level of augmentation ('none', 'light', 'medium', 'heavy')
            
        Returns:
            Augmented image
        """
        if augment_level == 'none':
            return img
        
        # Always apply horizontal flip
        img = self._apply_horizontal_flip(img)
        
        if augment_level == 'light':
            # Light augmentation
            img = self._apply_gaussian_blur(img, radius_range=(0, 0.7))
            img = self._apply_rotation(img, angle_range=( -10, 10))
        
        elif augment_level == 'medium':
            #Medium augmentation
            img = self._apply_gaussian_blur(img, radius_range=(0, 1.0))
            img = self._apply_rotation(img, angle_range=(-15, 15 ))
            img = self._apply_brightness_adjustment(img, factor_range=( 0.9, 1.1))
        
        elif augment_level == 'heavy':
            #Heavy augmentation
            img = self._apply_gaussian_blur(img, radius_range=( 0, 1.5 ))
            img = self._apply_rotation(img, angle_range=(-20, 20 ))
            img = self._apply_brightness_adjustment(img, factor_range=(0.85, 1.15))
            img = self._apply_contrast_adjustment(img, factor_range=(0.9, 1.1 ))
            img = self._apply_random_noise(img, intensity_range=(0, 8 ))
        
        return img
    
    def process_images(self, augment=True, val_split=0.2, augment_level = 'medium'):
        """
        Process all images in the dataset.
        
        Args:
            augment: Whether to apply augmentations
            val_split: Proportion of data to use for validation
            augment_level: Level of augmentation ('none', 'light', 'medium', 'heavy')
        """
        if self.train_df is None:
            self.load_metadata()
        
        #create directory structure
        self._create_directories()
        
        #split data into train and validation sets
        train_indices, val_indices = train_test_split(
            range(len(self.train_df)), 
            test_size=val_split, 
            stratify=self.train_df['Name'],
            random_state=42
        )
        
        #process training images
        print("Processing training images..." )
        self._process_subset(train_indices, "train", augment, augment_level )
        
        # Process validation images
        print("Processing validation images..." )
        self._process_subset(val_indices, "val ", False, 'none')
        
    def _process_subset(self, indices, subset_type, augment, augment_level):
        """
        Process a subset of images (train or validation)
        
        Args:
            indices: Indices of images to process
            subset_type: 'train' or 'val'
            augment: Whether to apply augmentations
            augment_level: Level of augmentation
        """
        for idx in tqdm(indices):
            row = self.train_df.iloc[ idx]
            uid = row[' SOPInstanceUID']
            label = row['Name']
            
            # Find DICOM file
            dicom_path = self.find_dicom_file(uid)
            if not dicom_path:
                continue
            
            try:
                #read DICOM
                dicom_data = pydicom.dcmread(dicom_path)
                
                #Process image
                img = self._normalize_dicom( dicom_data )
                img = self._resize_with_padding(img)
                
                #Save original image
                save_dir = os.path.join(self.output_path, subset_type, label)
                os.makedirs(save_dir, exist_ok=True)
                img.save(os.path.join(save_dir, f"{uid}.jpg"))
                
                # Apply augmentations
                if augment:
                    num_augmentations = 3  # Number of augmented versions to create
                    for i in range( num_augmentations):
                        aug_img = self._apply_augmentations( img, augment_level)
                        aug_img.save(os.path.join(save_dir, f"{uid}_aug{i}.jpg"))
            
            except Exception as e:
                print(f"Error processing {uid}: {str(e)}")
    
    def analyze_dataset( self):
        """
        Analyze the dataset distribution before and after processing
        
        Returns:
            DataFrame with distribution statistics
        """
        if self.train_df is None:
            self.load_metadata()
        
        original_dist = self.train_df['Name'].value_counts().sort_index()
        
        # Processed distribution
        processed_dist = {}
        for label in self.labels:
            train_path = os.path.join( self.output_path, "train", label )
            val_path = os.path.join(self.output_path , "val", label)
            
            train_count = len( os.listdir(train_path )) if os.path.exists(train_path ) else 0
            val_count = len(os.listdir(val_path )) if os.path.exists( val_path) else 0
            
            processed_dist[label ] = {
                "train": train_count,
                "val": val_count,
                "total": train_count + val_count 
            }
        
        #convert to DataFrame
        processed_df = pd.DataFrame.from_dict( processed_dist , orient = 'index' )
        
        #add original count
        for label in self.labels:
            if label in original_dist:
                processed_df.loc[ label, 'original' ] =  original_dist[ label ]
            else:
                processed_df.loc[label, 'original' ] = 0
        
        #calculate augmentation factor
        processed_df['aug_factor '] = processed_df[' total'] / processed_df[' original'].replace( 0, np.nan )
        
        return processed_df