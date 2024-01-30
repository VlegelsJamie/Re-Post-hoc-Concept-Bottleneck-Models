from glob import glob
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

from .constants import ISIC_DATA_DIR


class IsicDataset(Dataset):
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
    

def load_isic_data(preprocess, **kwargs):
    np.random.seed(kwargs['seed'])

    df = pd.read_csv(os.path.join(ISIC_DATA_DIR, 'isic_metadata.csv'))
    all_image_paths = glob(os.path.join(ISIC_DATA_DIR, '*.jpg'))
    id_to_path = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_paths}

    def path_getter(id):
        if id in id_to_path:
            return id_to_path[id] 
        else:
            return "-1"
        
    df['path'] = df['image_name'].map(path_getter)
    df = df[df.path != "-1"] 

    class_to_idx = {"benign": 0, "malignant": 1}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Splitting the dataset into benign and malignant
    df_benign = df[df['benign_malignant'] == 'benign']
    df_malignant = df[df['benign_malignant'] == 'malignant']

    # Sampling 1600 benign and 400 malignant for training
    df_benign_train = df_benign.sample(n=1600, random_state=kwargs['seed'])
    df_malignant_train = df_malignant.sample(n=400, random_state=kwargs['seed'])

    # Sampling 400 benign and 100 malignant for testing
    df_benign_test = df_benign.drop(df_benign_train.index).sample(n=400, random_state=kwargs['seed'])
    df_malignant_test = df_malignant.drop(df_malignant_train.index).sample(n=100, random_state=kwargs['seed'])

    # Combining the samples into train and test sets
    df_train = pd.concat([df_benign_train, df_malignant_train])
    df_val = pd.concat([df_benign_test, df_malignant_test])

    # Shuffling the datasets
    df_train = df_train.sample(frac=1, random_state=kwargs['seed']).reset_index(drop=True)
    df_val = df_val.sample(frac=1, random_state=kwargs['seed']).reset_index(drop=True)

    trainset = IsicDataset(df_train, preprocess)
    valset = IsicDataset(df_val, preprocess)
    print(f"Train, Val: {df_train.shape}, {df_val.shape}")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=kwargs['batch_size'],
                                              shuffle=True, num_workers=kwargs['num_workers'])
    
    val_loader = torch.utils.data.DataLoader(valset, batch_size=kwargs['batch_size'],
                                      shuffle=False, num_workers=kwargs['num_workers'])
    
    return train_loader, val_loader, idx_to_class
