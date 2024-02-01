import random
import json
import os
import numpy as np

from pcbm import get_concepts_multimodal, get_pcbm, get_pcbm_h

# Constants
DEVICE = "cuda"
NUM_WORKERS = 4
BATCH_SIZE = 64
CONCEPT_BANK_DIR = "trained_models/concept_banks/experiment_multimodal/"
PCBM_MODELS_DIR = "trained_models/pcbm_models/experiment_multimodal/"
PCBM_H_MODELS_DIR = "trained_models/pcbm_h_models/experiment_multimodal/"
NUM_SEEDS = 10
ALPHA = 0.99
LAM_VALUES = {"cifar10": 2, "cifar100": 2, "coco": 0.001}
DATASETS = ["cifar10", "cifar100", "coco"]

# Initialize test accuracy dictionaries
test_accs = {dataset: {"pcbm": [], "pcbm-h": []} for dataset in DATASETS}

random_seeds = [random.randint(0, 10000) for _ in range(NUM_SEEDS)]

for dataset in DATASETS:
    get_concepts_multimodal(out_dir=CONCEPT_BANK_DIR, 
                            classes=dataset, 
                            backbone_name="clip-RN50", 
                            device=DEVICE, 
                            recurse=1)
    
for seed in random_seeds:
    for dataset in DATASETS:
        concept_path = os.path.join(CONCEPT_BANK_DIR, f"multimodal_concept_clip-RN50_{dataset}_recurse:1.pkl")
        model_path = os.path.join(PCBM_MODELS_DIR, f"pcbm_{dataset}__clip-RN50__multimodal__lam-{LAM_VALUES[dataset]}__alpha-{ALPHA}__seed-{seed}.ckpt")

        run_info_pcbm = get_pcbm(
            baseline=False, 
            validation=False,
            concept_bank=concept_path, 
            out_dir=PCBM_MODELS_DIR, 
            dataset=dataset, 
            backbone_name="clip-RN50", 
            device=DEVICE, 
            seed=seed, 
            batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, 
            alpha=ALPHA, 
            lam=LAM_VALUES[dataset]
        )
        test_accs[dataset]["pcbm"].append(run_info_pcbm["test_acc"])

        run_info_pcbm_h = get_pcbm_h(
            out_dir=PCBM_H_MODELS_DIR, 
            pcbm_path=model_path, 
            concept_bank=concept_path, 
            device=DEVICE, 
            batch_size=BATCH_SIZE, 
            dataset=dataset, 
            seed=seed, 
            num_workers=NUM_WORKERS,
            num_epochs=10, 
            lr=0.01, 
            l2_penalty=LAM_VALUES[dataset]
            )
        test_accs[dataset]["pcbm-h"].append(run_info_pcbm_h["test_acc"])

# Calculate summary statistics
summary_stats = {dataset: {} for dataset in DATASETS}
for dataset in DATASETS:
    for model_type in ['pcbm', 'pcbm-h']:
        if test_accs[dataset][model_type]: 
            mean_acc = np.mean(test_accs[dataset][model_type])
            std_dev_acc = np.std(test_accs[dataset][model_type])
            summary_stats[dataset][model_type] = {'mean': mean_acc, 'std_dev': std_dev_acc}
        else:
            summary_stats[dataset][model_type] = {'mean': None, 'std_dev': None}

# Add summary statistics to test_accs dictionary
for dataset in DATASETS:
    test_accs[dataset]['summary'] = summary_stats[dataset]

os.makedirs("results/", exist_ok=True)
with open("results/test_accuracy_multimodal.json", 'w') as file:
    json.dump(test_accs, file, indent=4)

