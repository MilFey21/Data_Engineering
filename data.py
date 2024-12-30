import os
import io
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datasets import load_dataset, Dataset


class MRImerger:
    def __init__(self, data_url='Falah/Alzheimer_MRI'):
        self.data_url = data_url

    def download_data(self, data_url):
        # Download latest version
        data = load_dataset(data_url, split='train')
        return data

    def augment_images(self, class_label, class_data, target_class_count, datagen):
        current_count = len(class_data)
        needed_count = target_class_count - current_count
        augmented_images = []

        for index in range(current_count):
            img = class_data[index]['image']
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
                augmented_img_path = os.path.join('augmented_images', f'class_{class_label}_aug_{index}_{i}.jpg')
                tf.keras.preprocessing.image.save_img(augmented_img_path, batch[0])
                augmented_images.append({'image': augmented_img_path, 'label': class_label})
                needed_count -= 1

                if needed_count <= 0:
                    break

        return augmented_images

    def data_merge(self, data_url):
        # Download data
        data = self.download_data(data_url)

        # Set parameters
        target_class_count = 2500
        classes_to_augment = set()

        # Count images in each class
        class_counts = {label: 0 for label in set(data['label'])}
        for example in data:
            class_counts[example['label']] += 1

        # Identify classes that need augmentation
        classes_to_augment = {label for label, count in class_counts.items() if count < target_class_count}

        # Create a directory for augmented images if it doesn't exist
        os.makedirs('augmented_images', exist_ok=True)

        # Initialize ImageDataGenerator for augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Augment images for each class that needs it
        all_augmented_images = []
        for class_label in classes_to_augment:
            class_data = [example for example in data if example['label'] == class_label]
            all_augmented_images.extend(self.augment_images(class_label, class_data, target_class_count, datagen))

        # Create a new dataset from the augmented images
        #augmented_dataset = datasets.Dataset.from_list(all_augmented_images)

        # Create final merged dataset with exactly 2500 images per class
        final_data_list = []
        for label in set(data['label']):
            original_images = [example for example in data if example['label'] == label]
            augmented_images = [example for example in all_augmented_images if example['label'] == label]

            # Combine original and augmented images
            combined_images = original_images + augmented_images

            # Convert images to bytes
            for image_dict in combined_images:
                if isinstance(image_dict['image'], Image.Image):
                    image_bytes = io.BytesIO()
                    image_dict['image'].save(image_bytes, format='JPEG')  
                    image_dict['image'] = image_bytes.getvalue()  
                elif isinstance(image_dict['image'], str):
                    with open(image_dict['image'], 'rb') as f:
                        image_dict['image'] = f.read() 

            # If we have more than 2500, randomly sample to get exactly 2500
            if len(combined_images) > target_class_count:
                combined_images = np.random.choice(combined_images, target_class_count, replace=False).tolist()

            final_data_list.extend(combined_images)

        # Create final merged dataset with exactly 2500 images per class
        final_dataset_merged = Dataset.from_list(final_data_list)
        
        # Shuffle data
        final_dataset_merged = final_dataset_merged.shuffle(seed=42)

        return final_dataset_merged

###################################



# Create a directory for augmented images if it doesn't exist
#augmented_dir = 'augmented_images'
#os.makedirs(augmented_dir, exist_ok=True)


#data_url = 'Falah/Alzheimer_MRI'