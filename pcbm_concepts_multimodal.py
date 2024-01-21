import random
import json
import os

from original_code import get_concepts_multimodal, get_pcbm, get_pcbm_h


# Constants
DEVICE = "cuda"
NUM_WORKERS = 4
BATCH_SIZE = 64
CONCEPT_BANK_DIR = "concept_banks/"
PCBM_MODELS_DIR = "pcbm_models/"
PCBM_H_MODELS_DIR = "pcbm_h_models/"
NUM_SEEDS = 1
SEED_RANGE = (0, 10000)
ALPHA = 0.99
LAM_VALUES = {"cifar10": 2, "cifar100": 2, "coco-stuff": 0.001}
DATASETS = ["cifar10", "cifar100", "coco-stuff"]

# Initialize test accuracy dictionaries
test_accs = {dataset: {"pcbm": [], "pcbm-h": []} for dataset in DATASETS}


def learn_concepts_multimodal():
    for dataset in DATASETS:
        get_concepts_multimodal(out_dir=CONCEPT_BANK_DIR, 
                                classes=dataset, 
                                backbone_name="clip-RN50", 
                                device=DEVICE, 
                                recurse=1)


def train_pcbm(seed):
    for dataset in DATASETS:
        run_info_pcbm = get_pcbm(
            baseline=False, 
            concept_bank=f"{CONCEPT_BANK_DIR}/multimodal_concept_clip-RN50_{dataset}_recurse:1.pkl", 
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


def train_pcbm_h(seed):
    for dataset in DATASETS:
        pcbm_h_info = get_pcbm_h(
            out_dir=PCBM_H_MODELS_DIR, 
            pcbm_path=f"{PCBM_MODELS_DIR}/pcbm_{dataset}_clip-RN50_seed-{seed}.ckpt", 
            concept_bank=f"{CONCEPT_BANK_DIR}/multimodal_concept_clip-RN50_{dataset}_recurse:1.pkl", 
            device=DEVICE, 
            batch_size=BATCH_SIZE, 
            dataset=dataset, 
            seed=seed, 
            num_epochs=10, 
            lr=0.01, 
            l2_penalty=LAM_VALUES[dataset]
        )
        test_accs[dataset]["pcbm-h"].append(pcbm_h_info["accuracy"])


def save_results(results, file_path):
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)


def main():
    random_seeds = [random.randint(SEED_RANGE) for _ in range(NUM_SEEDS)]

    learn_concepts_multimodal()
    for seed in random_seeds:
        train_pcbm(seed)
        train_pcbm_h(seed)
    save_results(test_accs, "test_accuracy_results_multimodal_reproduction.json")


if __name__ == "__main__":
    main()
