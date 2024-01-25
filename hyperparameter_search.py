import random
import json
import os

from original_code import get_concepts_dataset, get_pcbm

# Constants
DEVICE = "cuda"
NUM_WORKERS = 4
BATCH_SIZE = 64
CONCEPT_BANK_DIR = "concept_banks/experiment_conceptdataset/"
BASELINE_DIR = "baseline_models/experiment_conceptdataset/"
PCBM_MODELS_DIR = "pcbm_models/experiment_conceptdataset/"
PCBM_H_MODELS_DIR = "pcbm_h_models/experiment_conceptdataset/"
VALIDATION_DIR = "validation_models/experiment_conceptdataset/"
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
N_SAMPLES = [20, 50, 100]
ALPHA = 0.99
LAM_SEARCH_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

# Fixed seed
SEED = 42

# Datasets and their configurations
DATASETS = {
    # "cifar10": {"backbone": "clip-RN50", "concept": "broden"},
    "cifar100": {"backbone": "clip-RN50", "concept": "broden"},
    # "coco-stuff": {"backbone": "clip-RN50", "concept": "broden"},
    "cub": {"backbone": "resnet18_cub", "concept": "cub"},
    "ham10000": {"backbone": "ham10000_inception", "concept": "derm7pt"},
    # "siim-isic": {"backbone": "ham10000_inception", "concept": "derm7pt"},
}

# Initialize test accuracy dictionaries
val_accs = {n_samples: {dataset: {"baseline": [], "pcbm": {lam: [] for lam in LAM_SEARCH_VALUES}, "pcbm-h": []} for dataset in DATASETS} for n_samples in N_SAMPLES}

for n_samples in N_SAMPLES:
    for dataset, config in DATASETS.items():
        concept_bank_path = os.path.join(CONCEPT_BANK_DIR, f"{config['concept']}_{config['backbone']}_{SEED}_{n_samples}.pkl")

        # Check if the concept bank file already exists
        if not os.path.exists(concept_bank_path):
            get_concepts_dataset(
                backbone_name=f"{config['backbone']}", 
                dataset_name=config['concept'], 
                out_dir=CONCEPT_BANK_DIR, 
                device=DEVICE, 
                seed=SEED, 
                num_workers=NUM_WORKERS, 
                batch_size=BATCH_SIZE, 
                C=C_VALUES, 
                n_samples=n_samples
            )

        for lam_value in LAM_SEARCH_VALUES:
            model_path = os.path.join(PCBM_MODELS_DIR, f"pcbm_{dataset}__{config['backbone']}__{config['concept']}__lam-{lam_value}__alpha-{ALPHA}__seed-{SEED}.ckpt")

            run_info_pcbm, run_info_baseline = get_pcbm(
                baseline=BASELINE_DIR, 
                validation=VALIDATION_DIR,
                concept_bank=concept_bank_path, 
                out_dir=PCBM_MODELS_DIR, 
                dataset=dataset, 
                backbone_name=config['backbone'], 
                device=DEVICE, 
                seed=SEED, 
                batch_size=BATCH_SIZE, 
                num_workers=NUM_WORKERS, 
                alpha=ALPHA, 
                lam=lam_value
            )
            val_accs[n_samples][dataset]["baseline"].append(run_info_baseline["test_acc"])
            val_accs[n_samples][dataset]["pcbm"][lam_value].append(run_info_pcbm["test_acc"])

os.makedirs("results/", exist_ok=True)
with open("results/hyperparameter_search.json", 'w') as file:
    json.dump(val_accs, file, indent=4)