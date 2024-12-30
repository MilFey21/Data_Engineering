import io
import os
from PIL import Image
import numpy as np
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import yaml
import random

class MRI_Dash:
    def __init__(self):
        """Initialize the MRI Dashboard"""
        pass

    # Class distribution
    def analyze_class_distribution(self, dataset, save_to_yaml='Dashboard/stats.yaml'):
        labels = [sample['label'] for sample in dataset]
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Create a bar plot for class distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=unique_labels, y=counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')

        # Save the plot as a PNG file
        plt.savefig('Dashboard/class_distribution.png')
        plt.close()

        # Save class distribution data to YAML file
        #class_distribution = dict(zip(unique_labels, counts))
        #with open(save_to_yaml, 'w') as yaml_file:
            #yaml.dump(class_distribution, yaml_file)

    
    # Sample Images
    def show_sample_images(self, dataset, num_samples=5, save_to_png='Dashboard/sample_images.png'):
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i in range(num_samples):
            # Check if 'image' is bytes and convert to PIL Image
            if isinstance(dataset[i]['image'], bytes):
                image = Image.open(io.BytesIO(dataset[i]['image']))
            # Check if 'image' is a file path and open it
            elif isinstance(dataset[i]['image'], str):
                image = Image.open(dataset[i]['image'])
            else: 
                image = dataset[i]['image']

            axes[i].imshow(image) 
            axes[i].set_title(f"Class: {dataset[i]['label']}")
            axes[i].axis('off')

        plt.savefig(save_to_png)
        plt.close() 
        
    # Dataset Correlation
    def analyze_dataset_correlations(self, image_list):
        """
        Compare all images in a dataset and plot histogram of correlations
        
        Parameters:
        image_list: List of images as numpy arrays
        
        Returns:
        correlations: List of correlation values
        fig: Matplotlib figure object
        """
        def compare_histograms(imageA, imageB):
            # Convert images to numpy arrays if they aren't already
            imageA = np.array(imageA)
            imageB = np.array(imageB)
            
            # Ensure images are grayscale
            if len(imageA.shape) == 3:
                imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            if len(imageB.shape) == 3:
                imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
                
            # Ensure images are in uint8 format
            imageA = np.uint8(imageA)
            imageB = np.uint8(imageB)

            # Calculate histograms
            histA = cv2.calcHist([imageA], [0], None, [256], [0, 256])
            histB = cv2.calcHist([imageB], [0], None, [256], [0, 256])

            # Normalize histograms
            histA = cv2.normalize(histA, histA).flatten()
            histB = cv2.normalize(histB, histB).flatten()

            # Compare using correlation
            correlation = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
            return correlation


        # Get 2 random indices
        num_samples = 100
        random_indices = random.sample(range(len(image_list)), num_samples)

        # Select samples using the random indices
        random_samples = [image_list[i] for i in random_indices]

        # Calculate correlations for all pairs
        correlations = []
        total_pairs = len(list(combinations(range(len(random_samples)), 2)))
        
        print(f"\nAnalyzing {len(random_samples)} images ({total_pairs} pairs)...")
        
        # Compare all pairs of images
        for i, j in tqdm(combinations(range(len(random_samples)), 2)):

        # Access the images and labels using the 'image' key
          imageA = np.array(Image.open(io.BytesIO(random_samples[i]['image'])))  
          imageB = np.array(Image.open(io.BytesIO(random_samples[j]['image']))) 
          
          correlation = compare_histograms(imageA, imageB)  
          correlations.append(correlation)
            
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.hist(correlations, bins=50, edgecolor='black')
        plt.title('Distribution of Image Correlations in Dataset')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Frequency')
        
        # Add statistical annotations
        mean_corr = np.mean(correlations)
        median_corr = np.median(correlations)
        std_corr = np.std(correlations)
        
        plt.axvline(mean_corr, color='r', linestyle='--', label=f'Mean: {mean_corr:.3f}')
        plt.axvline(median_corr, color='g', linestyle='--', label=f'Median: {median_corr:.3f}')
        
        plt.text(0.02, 0.95, f'Standard Deviation: {std_corr:.3f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot as a PNG file
        plt.savefig('Dashboard/correlation_distribution.png')
        plt.close()
    

    def dash_stats(self, dataset):
        self.analyze_class_distribution(dataset)
        self.show_sample_images(dataset)
        self.analyze_dataset_correlations(dataset)
       
        print('Stats updated')

    

    
        
        

