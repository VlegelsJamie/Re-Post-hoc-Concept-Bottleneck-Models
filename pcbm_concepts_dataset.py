import random
import json
import os

from original_code import get_concepts_dataset, get_pcbm, get_pcbm_h


# Constants
DEVICE = "cuda"
NUM_WORKERS = 4
BATCH_SIZE = 64
CONCEPT_BANK_DIR = "concept_banks/"
PCBM_MODELS_DIR = "pcbm_models/"
PCBM_H_MODELS_DIR = "pcbm_h_models/"
NUM_SEEDS = 1
SEED_RANGE = (0, 10000)
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
N_SAMPLES = 50
ALPHA = 0.99
LAM_VALUES = {"cifar10": 2.0, "cifar100": 2.0, "cub": 0.01, "ham10000": 2.0}

# Datasets and their configurations
DATASETS = {
    "cifar10": {"backbone": "clip-RN50", "concept": "broden"},
    "cifar100": {"backbone": "clip-RN50", "concept": "broden"},
    # "coco-stuff": {"backbone": "clip-RN50", "concept": "broden"},
    "cub": {"backbone": "resnet18_cub", "concept": "cub"},
    "ham10000": {"backbone": "ham10000_inception", "concept": "derm7pt"},
    # "siim-isic": {"backbone": "ham10000_inception", "concept": "derm7pt"},
}

# Initialize test accuracy dictionaries
test_accs = {dataset: {"baseline": [], "pcbm": [], "pcbm-h": []} for dataset in DATASETS}


def learn_concepts(seed):
    for dataset, config in DATASETS.items():
        concept_bank_path = f"{CONCEPT_BANK_DIR}/{config['concept']}_{config['backbone']}_{seed}_{N_SAMPLES}.pkl"

        # Check if the concept bank file already exists
        if not os.path.exists(concept_bank_path):
            get_concepts_dataset(
                backbone_name=f"{config['concept']}_{config['backbone']}", 
                dataset_name=config['concept'], 
                out_dir=CONCEPT_BANK_DIR, 
                device=DEVICE, 
                seed=seed, 
                num_workers=NUM_WORKERS, 
                batch_size=BATCH_SIZE, 
                C=C_VALUES, 
                n_samples=N_SAMPLES
            )


def train_baseline_and_pcbm(seed):
    for dataset, config in DATASETS.items():
        run_info_pcbm, run_info_baseline = get_pcbm(
            baseline=True, 
            concept_bank=f"{CONCEPT_BANK_DIR}/{config['concept']}_{config['backbone']}_{seed}_{N_SAMPLES}.pkl", 
            out_dir=PCBM_MODELS_DIR, 
            dataset=dataset, 
            backbone_name=config['backbone'], 
            device=DEVICE, 
            seed=seed, 
            batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, 
            alpha=ALPHA, 
            lam=LAM_VALUES[dataset]
        )
        test_accs[dataset]["baseline"].append(run_info_baseline["test_acc"])
        test_accs[dataset]["pcbm"].append(run_info_pcbm["test_acc"])


def train_pcbm_h(seed):
    for dataset, config in DATASETS.items():
        pcbm_h_accuracy = get_pcbm_h(
            out_dir=PCBM_H_MODELS_DIR,
            pcbm_path=f"{PCBM_MODELS_DIR}/pcbm_{dataset}__{config['backbone']}__{config['concept']}__lam-{LAM_VALUES[dataset]}__alpha-{ALPHA}__seed-{seed}.ckpt",
            concept_bank=f"{CONCEPT_BANK_DIR}/{config['concept']}_{config['backbone']}_{seed}_{N_SAMPLES}.pkl",
            device=DEVICE,
            batch_size=BATCH_SIZE,
            dataset=dataset,
            seed=seed,
            num_epochs=10,
            lr=0.01,
            l2_penalty=0.01
        )["accuracy"]
        test_accs[dataset]["pcbm-h"].append(pcbm_h_accuracy)


def save_results(results, file_path):
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)


def main():
    random_seeds = [random.randint(SEED_RANGE) for _ in range(NUM_SEEDS)]

    for seed in random_seeds:
        learn_concepts(seed)
        train_baseline_and_pcbm(seed)
        train_pcbm_h(seed)
    
    # Save results to a JSON file
    save_results(test_accs, "results/test_accuracy_results_reproduction.json")
        

if __name__ == "__main__":
    main()
