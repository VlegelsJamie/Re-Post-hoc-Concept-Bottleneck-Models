import os
import pickle
import torch
import argparse

import numpy as np

from .models import get_model
from .concepts import learn_concept_bank
from .data import get_concept_loaders


def get_concepts_dataset(**kwargs):
    n_samples = kwargs['n_samples']

    # Bottleneck part of model
    backbone, preprocess = get_model(**kwargs)
    backbone = backbone.to(kwargs['device'])
    backbone = backbone.eval()
    
    concept_libs = {C: {} for C in kwargs['C']}
    # Get the positive and negative loaders for each concept. 
    
    concept_loaders = get_concept_loaders(kwargs['dataset_name'], preprocess, n_samples=n_samples, batch_size=kwargs['batch_size'], 
                                          num_workers=kwargs['num_workers'], seed=kwargs['seed'])
    
    np.random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
    for concept_name, loaders in concept_loaders.items():
        pos_loader, neg_loader = loaders['pos'], loaders['neg']
        # Get CAV for each concept using positive/negative image split
        cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, n_samples, kwargs['C'], device=kwargs['device'])
        
        # Store CAV train acc, val acc, margin info for each regularization parameter and each concept
        for C in kwargs['C']:
            concept_libs[C][concept_name] = cav_info[C]
            print(concept_name, C, cav_info[C][1], cav_info[C][2])

    # Save CAV results    
    for C in concept_libs.keys():
        lib_path = os.path.join(kwargs['out_dir'], f"{kwargs['dataset_name']}_{kwargs['backbone_name']}_{kwargs['seed']}_{n_samples}.pkl")
        with open(lib_path, "wb") as f:
            pickle.dump(concept_libs[C], f)
        print(f"Saved to: {lib_path}")        
    
        total_concepts = len(concept_libs[C].keys())
        print(f"File: {lib_path}, Total: {total_concepts}")
