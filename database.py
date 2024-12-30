import pickle
import os
from PIL import Image
import sqlite3
import io
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm


class TensorDatabaseManager:
    def __init__(self, db_path='tensor_dataset.db'):
        self.db_path = db_path
        self.initialize_database()

    def initialize_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Create the dataset_metadata table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''') 
            # Create the tensor_data table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tensor_data BLOB,
                    label INTEGER,
                    shape TEXT,
                    dtype TEXT
                )
            ''')
            conn.commit()

    def tensor_to_blob(self, tensor):
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()

    def blob_to_tensor(self, blob):
        buffer = io.BytesIO(blob)
        return torch.load(buffer)

    def save_dataset(self, dataset, batch_size=1000):
        """
        Save entire dataset to database

        Args:
            dataset: PyTorch Dataset object
            batch_size: Number of samples to save in each batch
        """
        print("Saving dataset to database...")
        total_samples = len(dataset)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            sample_image = dataset[0]['image']  # Get the PIL image
            
            # Convert bytes to PIL Image
            sample_image = Image.open(io.BytesIO(sample_image))
            
            sample_tensor = transforms.ToTensor()(sample_image)

            metadata = {
                'total_samples': total_samples,
                'tensor_shape': str(sample_tensor.shape),
                'tensor_dtype': str(sample_tensor.dtype)
            }


            # Save tensors in batches
            for i in tqdm(range(0, total_samples, batch_size)):
                batch_data = []
                end_idx = min(i + batch_size, total_samples)

                for j in range(i, end_idx):
                    # Get the image and label from the dataset
                    image, label = dataset[j]['image'], dataset[j]['label']

                    # Convert the bytes image data to a PIL Image
                    image = Image.open(io.BytesIO(image)) 

                    # Convert the PIL Image to a PyTorch tensor
                    tensor = transforms.ToTensor()(image)

                    blob = self.tensor_to_blob(tensor)
                    shape = str(tensor.shape)
                    dtype = str(tensor.dtype)
                    batch_data.append((blob, label, shape, dtype))

                #cursor.executemany( , batch_data)

                conn.commit()

        print(f"Successfully saved {total_samples} samples to database")

    def load_dataset(self, batch_size=1000):
        """
        Load dataset from database

        Returns:
            LoadedTensorDataset object
        """
        print("Loading dataset from database...")

        tensors = []
        labels = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get total count
            cursor.execute('SELECT COUNT(*) FROM tensor_data')
            total_samples = cursor.fetchone()[0]

            # Load data in batches
            for offset in tqdm(range(0, total_samples, batch_size)):
                cursor.execute(' SELECT tensor_data, label FROM tensor_data', (batch_size, offset))

                batch_data = cursor.fetchall()

                for blob, label in batch_data:
                    tensor = self.blob_to_tensor(blob)
                    tensors.append(tensor)
                    labels.append(label)

        return LoadedTensorDataset(tensors, labels)

    def get_metadata(self):
        """Retrieve dataset metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM dataset_metadata')
            return dict(cursor.fetchall())

def save_tensor_dataset_to_db(dataset, db_path='tensor_dataset.db'):
    # Initialize database manager
    db_manager = TensorDatabaseManager(db_path)

    # Save dataset
    db_manager.save_dataset(dataset)

    # Print metadata
    metadata = db_manager.get_metadata()
    print("\nDataset metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")



class LoadedTensorDataset(Dataset):
    """Dataset class for loaded tensor data"""
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]


# To save the dataset:
# save_tensor_dataset_to_db(final_dataset_merged)


# To load the dataset:
# db_manager = TensorDatabaseManager()
# loaded_dataset = db_manager.load_dataset()