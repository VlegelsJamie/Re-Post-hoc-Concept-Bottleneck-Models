from glob import glob
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from .constants import COCO_DATA_DIR


class CocoDataset(Dataset):
    def __init__(self, df, preprocess=None):
        self.df = df
        self.preprocess = preprocess
    
    def __len__(self):
        return(len(self.df))
    
    def __getitem__(self, index):
        X = Image.open(self.df['path'].iloc[index])
        y = torch.tensor(int(self.df['target'].iloc[index]))
        if self.preprocess:
            X = self.preprocess(X)
        return X,y
    

def load_coco_data(preprocess, **kwargs):
    np.random.seed(kwargs['seed'])

    df = pd.read_csv(os.path.join(COCO_DATA_DIR, 'isic_metadata.csv'))
    all_image_paths = glob(os.path.join(COCO_DATA_DIR, '*', '*.jpg'))
    id_to_path = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_paths}

    def path_getter(id):
        if id in id_to_path:
            return id_to_path[id] 
        else:
            return "-1"
        
    df['path'] = df['image_id'].map(path_getter)
    df = df[df.path != "-1"] 
    class_to_idx = {"benign": 0, "malignant": 1}

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    _, df_val = train_test_split(df, test_size=0.20, random_state=kwargs['seed'], stratify=df["dx"])
    df_train = df[~df.image_id.isin(df_val.image_id)]
    trainset = CocoDataset(df_train, preprocess)
    valset = CocoDataset(df_val, preprocess)
    print(f"Train, Val: {df_train.shape}, {df_val.shape}")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=kwargs['batch_size'],
                                              shuffle=True, num_workers=kwargs['num_workers'])
    
    val_loader = torch.utils.data.DataLoader(valset, batch_size=kwargs['batch_size'],
                                      shuffle=False, num_workers=kwargs['num_workers'])
    
    return train_loader, val_loader, idx_to_class