import random

import numpy as np
import cv2
from PIL import Image
from PIL import ImageFilter as Filter
import torch


def edge_enhancing(array):
    method = np.random.choice(['ada_thold', 'laplacian', 'edge_enahnced'])
    
    if method=='ada_thold':     
        return np.expand_dims(cv2.adaptiveThreshold(array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1), 2)
    
    elif method=='laplacian':
        return np.expand_dims(cv2.Laplacian(array,cv2.CV_64F, ksize=5), 2)
    
    else:
        image = Image.fromarray(np.squeeze(array, axis=2)).convert('L')
        return np.expand_dims(np.asarray(image.filter(Filter.EDGE_ENHANCE_MORE)), 2)

def de_texturization(array):
    n = np.random.choice([5, 9, 13, 15])
    sigma = np.random.choice([50, 65, 75])
    
    return np.expand_dims(cv2.bilateralFilter(array, n, sigma, sigma), 2)

def tumbnail(array, shape=(512,512)):
    return cv2.resize(array, shape) 

def random_crop(array):
    method = np.random.choice(['left', 'right', 'top', 'down'])
    v_center = array.shape[1]//2
    h_center = array.shape[0]//2
    
    if method == 'left':
        return array[:,:v_center,:]
    elif method == 'right':
        return array[:,v_center:,:]
    elif method == 'top':
        return array[:h_center,:,:]
    elif method == 'down':
        return array[h_center:,:,:]
    else:
        return array

def pre_processing(image, input_size=224):
    img = np.array(image)
    
    # Optional de-texturize and crop
    if np.random.rand() < 0.5:
        img = de_texturization(img)
        img = edge_enhancing(img)
    
    img = tumbnail(img, (input_size, input_size))
    
    # Convert to 3 channels if grayscale
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)  # make it (H, W, 1)
        img = np.repeat(img, 3, axis=2)    # then to (H, W, 3)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    
    img = img.astype(np.float32) / 255.0  # normalize to [0,1]
    img = np.transpose(img, (2, 0, 1))     # to (C, H, W)
    return torch.tensor(img)
