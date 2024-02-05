from glob import glob
import os
import random
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from .constants import METASHIFT_DATA_DIR


SCENARIOS = {0: {'train': 'bed(dog)', 'test': 'bed(cat)'},
             1: {'train': 'bed(cat)', 'test': 'bed(dog)'},
             2: {'train': 'table(dog)', 'test': 'table(cat)'},
             3: {'train': 'table(cat)', 'test': 'table(dog)'},
             4: {'train': 'table(books)', 'test': 'table(dog)'},
             5: {'train': 'table(books)', 'test': 'table(cat)'},
             6: {'train': 'car(dog)', 'test': 'car(cat)'},
             7: {'train': 'car(cat)', 'test': 'car(dog)'},
             8: {'train': 'cow(dog)', 'test': 'cow(cat)'},
             9: {'train': 'keyboard(dog)', 'test': 'keyboard(cat)'},
}


class MetashiftDataset(Dataset):
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
    

def load_images_from_dir(directory, label, n_samples=50):
    """ Load n_samples images from a directory and assign them the given label. """
    all_files = glob(os.path.join(directory, '*.jpg'))  # Assuming images are in jpg format
    sampled_files = random.sample(all_files, n_samples)
    return pd.DataFrame({'path': sampled_files, 'target': [label]*n_samples})
    

def load_metashift_data(preprocess, scenario, **kwargs):
    np.random.seed(kwargs['seed'])
    random.seed(kwargs['seed'])

    # Define task classes and scenario specific classes
    if scenario in [0, 1, 6, 7, 8, 9]:
        task = ["airplane", "bed", "car", "cow", "keyboard"]
    else:
        task = ["beach", "computer", "motorcycle", "stove", "table"]
    
    spurious_train_class, spurious_test_class = SCENARIOS[scenario]['train'], SCENARIOS[scenario]['test']
    
    # Load images for each class
    df_list_train = []
    df_list_val = []
    for i, cls in enumerate(task):
        directory = os.path.join(METASHIFT_DATA_DIR, cls)
        directory_spurious_train = os.path.join(METASHIFT_DATA_DIR, spurious_train_class)
        directory_spurious_test = os.path.join(METASHIFT_DATA_DIR, spurious_test_class)
        if cls == spurious_train_class.split('(')[0]:
            df_list_train.append(load_images_from_dir(directory_spurious_train, label=i, n_samples=50))
            df_list_val.append(load_images_from_dir(directory_spurious_test, label=i, n_samples=50))
        else:
            df_list_train.append(load_images_from_dir(directory, label=i, n_samples=50))
            df_list_val.append(load_images_from_dir(directory, label=i, n_samples=50))

    # Combine and shuffle datasets
    df_train = pd.concat(df_list_train).sample(frac=1).reset_index(drop=True)
    df_val = pd.concat(df_list_val).sample(frac=1).reset_index(drop=True)

    print(df_train)

    # Index to class mapping
    idx_to_class = {i: cls for i, cls in enumerate(task)}

    trainset = MetashiftDataset(df_train, preprocess)
    valset = MetashiftDataset(df_val, preprocess)
    print(f"Train, Val: {df_train.shape}, {df_val.shape}")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=kwargs['batch_size'],
                                              shuffle=True, num_workers=kwargs['num_workers'])
    
    val_loader = torch.utils.data.DataLoader(valset, batch_size=kwargs['batch_size'],
                                      shuffle=False, num_workers=kwargs['num_workers'])
    
    return train_loader, val_loader, idx_to_class