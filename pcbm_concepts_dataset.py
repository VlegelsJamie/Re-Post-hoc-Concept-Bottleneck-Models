import random
import json
import os
import numpy as np

from pcbm import get_concepts_dataset, get_pcbm, get_pcbm_h

if __name__ == '__main__':
    # Constants
    DEVICE = "cuda"
    NUM_WORKERS = 4
    BATCH_SIZE = 64
    CONCEPT_BANK_DIR = "trained_models/concept_banks/experiment_conceptdataset/"
    BASELINE_DIR = "trained_models/baseline_models/experiment_conceptdataset/"
    PCBM_MODELS_DIR = "trained_models/pcbm_models/experiment_conceptdataset/"
    PCBM_H_MODELS_DIR = "trained_models/pcbm_h_models/experiment_conceptdataset/"
    NUM_SEEDS = 10
    C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
    N_SAMPLES = 50
    ALPHA = 0.99
    LAM_VALUES = {
        "cifar10": {"baseline": 1e-7, "pcbm": 2.0},
        "cifar100": {"baseline": 1e-7, "pcbm": 2.0},
        "coco": {"baseline": 1e-7, "pcbm": 0.001},
        "cub": {"baseline": None, "pcbm": 0.01},
        "ham10000": {"baseline": None, "pcbm": 2.0},
        "isic": {"baseline": 1e-7, "pcbm": 0.001}
    }

    # Datasets and their configurations
    DATASETS = {
        #"cifar10": {"backbone": "clip-RN50", "concept": "broden"},
        #"cifar100": {"backbone": "clip-RN50", "concept": "broden"},
        "coco": {"backbone": "clip-RN50", "concept": "broden"},
        #"cub": {"backbone": "resnet18_cub", "concept": "cub"},
        #"ham10000": {"backbone": "ham10000_inception", "concept": "derm7pt"},
        #"isic": {"backbone": "ham10000_inception", "concept": "derm7pt"},
    }

    # Initialize test accuracy dictionaries
    test_accs = {dataset: {"baseline": [], "pcbm": [], "pcbm-h": []} for dataset in DATASETS}

    random_seeds = [random.randint(0, 10000) for _ in range(NUM_SEEDS)]
    random_seeds = [42]

    for seed in random_seeds:
        for dataset, config in DATASETS.items():
            concept_bank_path = os.path.join(CONCEPT_BANK_DIR, f"{config['concept']}_{config['backbone']}_{seed}_{N_SAMPLES}.pkl")
            model_path = os.path.join(PCBM_MODELS_DIR, f"pcbm_{dataset}__{config['backbone']}__{config['concept']}__lam-{LAM_VALUES[dataset]['pcbm']}__alpha-{ALPHA}__seed-{seed}.ckpt")

            # Check if the concept bank file already exists
            if not os.path.exists(concept_bank_path):
                get_concepts_dataset(
                    backbone_name=config['backbone'], 
                    dataset_name=config['concept'], 
                    out_dir=CONCEPT_BANK_DIR, 
                    device=DEVICE, 
                    seed=seed, 
                    num_workers=NUM_WORKERS, 
                    batch_size=BATCH_SIZE, 
                    C=C_VALUES, 
                    n_samples=N_SAMPLES
                )

            print(LAM_VALUES[dataset]["pcbm"], LAM_VALUES[dataset]["baseline"])

            run_info_pcbm = get_pcbm(
                baseline=False,
                validation=False, 
                concept_bank=concept_bank_path, 
                out_dir=PCBM_MODELS_DIR, 
                dataset=dataset, 
                backbone_name=config['backbone'], 
                device=DEVICE, 
                seed=seed, 
                batch_size=BATCH_SIZE, 
                num_workers=NUM_WORKERS, 
                alpha=ALPHA, 
                lam=LAM_VALUES[dataset]["pcbm"],
                lam_baseline=LAM_VALUES[dataset]["baseline"]
            )
            #test_accs[dataset]["baseline"].append(run_info_baseline["test_acc"])
            test_accs[dataset]["pcbm"].append(run_info_pcbm["test_acc"])

            """
            run_info_pcbm_h = get_pcbm_h(
                out_dir=PCBM_H_MODELS_DIR,
                pcbm_path=model_path,
                concept_bank=concept_bank_path,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                dataset=dataset,
                seed=seed,
                num_workers=NUM_WORKERS,
                num_epochs=10,
                lr=0.01,
                l2_penalty=0.01
            )
            test_accs[dataset]["pcbm-h"].append(run_info_pcbm_h["test_acc"])
            """

    # Calculate summary statistics
    summary_stats = {dataset: {} for dataset in DATASETS}
    for dataset in DATASETS:
        for model_type in ['baseline', 'pcbm', 'pcbm-h']:
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
    with open("results/test_accuracy_feature_attribution.json", 'w') as file:
        json.dump(test_accs, file, indent=4)