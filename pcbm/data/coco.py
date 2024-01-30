from glob import glob
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import json
from typing import Dict

from .constants import COCO_IMAGES, COCO_META, COCO_LABELS, COCO_ANNOTATIONS


target_classes = {"ids": [47, 46, 31, 53, 3, 6, 64, 50, 78, 76, 35, 85, 37, 75, 36, 80, 89, 43, 41, 40], 
                  "names": ["cup", "wine glass", "handbag", "apple", "car", "bus", "potted plant", 
                            "spoon", "microwave", "keyboard", "skis", "clock", "sports ball", "remote", 
                            "snowboard", "toaster", "hair drier", "tennis racket", "skateboard", "baseball glove"]}


def load_json(json_path):
    with open(json_path, "r") as rf:
        return json.load(rf)


class CocoDataset(Dataset):
    def __init__(self, image_data, preprocess=None):
        self.image_data = image_data
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, index):
        data = self.image_data[index]
        X = Image.open(data['path'])
        y = torch.tensor(data['target'])
        if self.preprocess:
            X = self.preprocess(X)
        return X, y


def process_images(image_folder_path, annotations_folder_path, target_classes):
    image_data = []
    for image_file in os.listdir(annotations_folder_path):
        if image_file.endswith('.png'):
            annotations_path = os.path.join(annotations_folder_path, image_file)
            image_path = os.path.join(image_folder_path, os.path.splitext(image_file)[0] + '.jpg')

            image = Image.open(annotations_path)
            image = np.array(image)

            # Find unique labels in the image
            unique_labels = np.unique(image)
            target_labels = [label for label in unique_labels if label in target_classes['ids']]

            # One-hot encode the labels
            encoded_labels = [1 if label in target_labels else 0 for label in target_classes['ids']]

            image_data.append({'path': image_path, 'target': encoded_labels})
    return image_data
    

def load_coco_data(preprocess, **kwargs):
    np.random.seed(kwargs['seed'])

    class_to_idx = {"cup": 0, "wine glass": 1, "handbag": 2, "apple": 3, "car": 4, "bus": 5, "potted plant": 6,
                    "spoon": 7, "microwave": 8, "keyboard": 9, "skis": 10, "clock": 11, "sports ball": 12,
                    "remote": 13, "snowboard": 14, "toaster": 15, "hair drier": 16, "tennis racket": 17,
                    "skateboard": 18, "baseball glove": 19}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    processed_data_file = "trained_models/labels_coco.json"

    # Check if processed data file exists
    if os.path.exists(processed_data_file):
        with open(processed_data_file, 'r') as rf:
            image_data = json.load(rf)
    else:
        os.makedirs("trained_models/", exist_ok=True)
        image_data = process_images(COCO_IMAGES, COCO_ANNOTATIONS, target_classes)
        with open(processed_data_file, 'w') as wf:
            json.dump(image_data, wf)

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(image_data, test_size=0.33, random_state=kwargs.get('seed', 42))

    # Create Datasets
    train_dataset = CocoDataset(train_data, preprocess=preprocess)
    val_dataset = CocoDataset(val_data, preprocess=preprocess)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=kwargs['batch_size'],
                                               shuffle=True, num_workers=kwargs['num_workers'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=kwargs['batch_size'],
                                             shuffle=False, num_workers=kwargs['num_workers'])

    return train_loader, val_loader, idx_to_class