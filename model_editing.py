import random
import json
import os
import numpy as np

from pcbm import get_concepts_dataset, get_pcbm, get_pcbm_h


def write_statistics(accs, filename):
    # Calculate summary statistics
    summary_stats = {scenario: {} for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        for model_type in ['pcbm', 'pcbm-h']:
            if accs[scenario][model_type]: 
                mean_acc = np.mean(accs[scenario][model_type])
                std_dev_acc = np.std(accs[scenario][model_type])
                summary_stats[scenario][model_type] = {'mean': mean_acc, 'std_dev': std_dev_acc}
            else:
                summary_stats[scenario][model_type] = {'mean': None, 'std_dev': None}

    # Add summary statistics to test_accs dictionary
    for scenario in SCENARIOS:
        accs[scenario]['summary'] = summary_stats[scenario]

    filename = "results/" + filename
    os.makedirs("results/", exist_ok=True)
    with open(filename, 'w') as file:
        json.dump(accs, file, indent=4)


# Constants
DEVICE = "cuda"
NUM_WORKERS = 4
BATCH_SIZE = 64
CONCEPT_BANK_DIR = "trained_models/concept_banks/experiment_editing/"
PCBM_MODELS_DIR = "trained_models/pcbm_models/experiment_editing/"
PCBM_H_MODELS_DIR = "trained_models/pcbm_h_models/experiment_editing/"
NUM_SEEDS = 1
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
N_SAMPLES = 100
ALPHA = 0.99
LAM_VALUE = 0.002

SCENARIOS = {1: {'train': 'bed(dog)', 'test': 'bed(cat)'},
             2: {'train': 'bed(cat)', 'test': 'bed(dog)'},
             3: {'train': 'table(dog)', 'test': 'table(cat)'},
             4: {'train': 'table(cat)', 'test': 'table(dog)'},
             5: {'train': 'table(books)', 'test': 'table(dog)'},
             6: {'train': 'table(books)', 'test': 'table(cat)'},
             7: {'train': 'car(dog)', 'test': 'car(cat)'},
             8: {'train': 'car(cat)', 'test': 'car(dog)'},
             9: {'train': 'cow(dog)', 'test': 'cow(cat)'},
             10: {'train': 'keyboard(dog)', 'test': 'keyboard(cat)'},
}

# Datasets and their configurations
DATASET = "metashift"
BACKBONE = "resnet50"
CONCEPT = "broden"

# Initialize test accuracy dictionaries
unedited = {scenario: {"pcbm": [], "pcbm-h": []} for scenario in SCENARIOS}
prune = {scenario: {"pcbm": [], "pcbm-h": []} for scenario in SCENARIOS}
prune_normalize = {scenario: {"pcbm": [], "pcbm-h": []} for scenario in SCENARIOS}
fine_tune = {scenario: {"pcbm": [], "pcbm-h": []} for scenario in SCENARIOS}

random_seeds = [random.randint(0, 10000) for _ in range(NUM_SEEDS)]

for seed in random_seeds:
    concept_bank_path = os.path.join(CONCEPT_BANK_DIR, f"{CONCEPT}_{BACKBONE}_{seed}_{N_SAMPLES}.pkl")

    # Check if the concept bank file already exists
    if not os.path.exists(concept_bank_path):
        get_concepts_dataset(
            backbone_name=BACKBONE, 
            dataset_name=CONCEPT, 
            out_dir=CONCEPT_BANK_DIR, 
            device=DEVICE, 
            seed=seed, 
            num_workers=NUM_WORKERS, 
            batch_size=BATCH_SIZE, 
            C=C_VALUES, 
            n_samples=N_SAMPLES
        )
    
    for scenario, scenario_config in SCENARIOS.items():
        model_path = os.path.join(PCBM_MODELS_DIR, f"pcbm_{DATASET + '_' + str(scenario)}__{BACKBONE}__{CONCEPT}__lam-{LAM_VALUE}__alpha-{ALPHA}__seed-{seed}.ckpt")

        run_info_pcbm = get_pcbm(
            baseline=False,
            validation=False, 
            concept_bank=concept_bank_path, 
            out_dir=PCBM_MODELS_DIR, 
            dataset=DATASET + "_" + str(scenario), 
            backbone_name=BACKBONE, 
            device=DEVICE, 
            seed=seed, 
            batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, 
            alpha=ALPHA, 
            lam=LAM_VALUE
        )
        unedited[scenario]["pcbm"].append(run_info_pcbm["test_acc"])

        run_info_pcbm_h = get_pcbm_h(
            out_dir=PCBM_H_MODELS_DIR,
            pcbm_path=model_path,
            concept_bank=concept_bank_path,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            dataset=DATASET + "_" + str(scenario),
            seed=seed,
            num_workers=NUM_WORKERS,
            num_epochs=10,
            lr=0.01,
            l2_penalty=0.01
        )
        unedited[scenario]["pcbm-h"].append(run_info_pcbm_h["test_acc"])

# Prune
        
# Prune + normalize
        
# Fine-tune

write_statistics(unedited, "test_accuracy_unedited.json")
write_statistics(prune, "test_accuracy_prune.json")
write_statistics(prune_normalize, "test_accuracy_prune_normalize.json")
write_statistics(fine_tune, "test_accuracy_fine_tune.json")

    
