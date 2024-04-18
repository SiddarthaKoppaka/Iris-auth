from PIL import Image
import os
import numpy as np
from datasets import Dataset
import torch

def import_images(root_dir, target_size=(192, 192)):
    """
    Import images from the specified root directory and create a Hugging Face Dataset.
    Cross Folding Technique implemented with a binary mask.
    """
    file_paths = []
    images = []
    labels = []

    for label_folder in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label_folder)

        if os.path.isdir(label_path):
            left_folder = os.path.join(label_path, 'left')
            right_folder = os.path.join(label_path, 'right')

            if os.path.exists(left_folder) and os.path.exists(right_folder):
                left_files = [file for file in os.listdir(left_folder) if file.lower().endswith('.bmp')]
                right_files = [file for file in os.listdir(right_folder) if file.lower().endswith('.bmp')]

                for left_file, right_file in zip(left_files, right_files):
                    left_image_path = os.path.join(left_folder, left_file)
                    right_image_path = os.path.join(right_folder, right_file)

                    left_image = Image.open(left_image_path).resize(target_size)
                    right_image = Image.open(right_image_path).resize(target_size)

                    # Convert images to arrays for processing
                    left_array = np.array(left_image)
                    right_array = np.array(right_image)

                    # Create a binary mask
                    mask = np.random.choice([0, 1], size=target_size, p=[0.5, 0.5]).astype(np.bool_)
                    complement_mask = np.invert(mask)

                    # Apply masks to images
                    crossfolded_array = np.where(mask[..., None], left_array, right_array)

                    # Convert array back to an image
                    crossfolded_image = Image.fromarray(crossfolded_array)

                    file_paths.append(f'{label_folder}_crossfolded.jpg')
                    images.append(crossfolded_image)
                    labels.append(label_folder)

    dataset_dict = {
        "image": images,
        "labels": labels,
    }

    dataset = Dataset.from_dict(dataset_dict)

    return dataset

def labels(dataset):
    # Checking for number of classes
    unique_labels = []
    for x in dataset['labels']:
        if x not in unique_labels:
            unique_labels.append(x)
    print(unique_labels)
    return unique_labels



def collate_fn(batch, unique_labels):
    label_encoder = {label: index for index, label in enumerate(unique_labels)}  # unique_labels should contain all unique label strings in your dataset

    # Convert string labels to numerical format using label_encoder
    numerical_labels = [label_encoder[x['labels']] for x in batch]

    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor(numerical_labels)
    }