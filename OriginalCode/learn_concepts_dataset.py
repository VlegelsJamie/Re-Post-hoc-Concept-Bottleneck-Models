import os
import pickle
import torch
import argparse

import numpy as np

from .models import get_model
from .concepts import learn_concept_bank
from .data import get_concept_loaders


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--dataset-name", default="cub", type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--num-workers", default=4, type=int, help="Number of workers in the data loader.")
    parser.add_argument("--batch-size", default=100, type=int, help="Batch size in the concept loader.")
    parser.add_argument("--C", nargs="+", default=[0.01, 0.1], type=float, help="Regularization parameter for SVMs.")
    parser.add_argument("--n-samples", default=50, type=int, 
                        help="Number of positive/negative samples used to learn concepts.")
    return parser.parse_args()


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
        lib_path = os.path.join(kwargs['out_dir'], f"{kwargs['dataset_name']}_{kwargs['backbone_name']}_{C}_{n_samples}.pkl")
        with open(lib_path, "wb") as f:
            pickle.dump(concept_libs[C], f)
        print(f"Saved to: {lib_path}")        
    
        total_concepts = len(concept_libs[C].keys())
        print(f"File: {lib_path}, Total: {total_concepts}")


def main():
    args = config()
    get_concepts_dataset(backbone_name=args.backbone_name, dataset_name=args.dataset_name, out_dir=args.out_dir, 
                           device=args.device, seed=args.seed, num_workers=args.num_workers, batch_size=args.batch_size,
                           C=args.C, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
