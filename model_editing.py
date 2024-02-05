import json
import os
import numpy as np
import torch

from pcbm import get_concepts_dataset, get_pcbm, get_pcbm_h
from pcbm.models.pcbm_utils import PosthocHybridCBM
from pcbm.training_tools.embedding_tools import load_or_compute_projections


def write_statistics(accs, filename):
    # Calculate summary statistics
    summary_stats = {scenario: {} for scenario in range(10)}
    for scenario in range(10):
        for model_type in ['pcbm', 'pcbm-h']:
            if accs[scenario][model_type]: 
                mean_acc = np.mean(accs[scenario][model_type])
                std_dev_acc = np.std(accs[scenario][model_type])
                summary_stats[scenario][model_type] = {'mean': mean_acc, 'std_dev': std_dev_acc}
            else:
                summary_stats[scenario][model_type] = {'mean': None, 'std_dev': None}

    # Add summary statistics to test_accs dictionary
    for scenario in range(10):
        accs[scenario]['summary'] = summary_stats[scenario]

    filename = "results/" + filename
    os.makedirs("results/", exist_ok=True)
    with open(filename, 'w') as file:
        json.dump(accs, file, indent=4)


# Constants
DEVICE = "cuda"
NUM_WORKERS = 4
BATCH_SIZE = 64
SEED = 42
CONCEPT_BANK_DIR = "trained_models/concept_banks/experiment_editing/"
PCBM_MODELS_DIR = "trained_models/pcbm_models/experiment_editing/"
PCBM_H_MODELS_DIR = "trained_models/pcbm_h_models/experiment_editing/"
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
N_SAMPLES = 100
ALPHA = 0.99
LAM_VALUE = 0.002

# Datasets and their configurations
DATASET = "metashift"
BACKBONE = "resnet50"
CONCEPT = "broden"

SCENARIOS = {0: {'class': 'bed', 'prune': 'dog'},
             1: {'train': 'bed', 'test': 'cat'},
             2: {'train': 'table', 'test': 'dog'},
             3: {'train': 'table', 'test': 'cat'},
             4: {'train': 'table', 'test': 'books'},
             5: {'train': 'table', 'test': 'books'},
             6: {'train': 'car', 'test': 'dog'},
             7: {'train': 'car', 'test': 'cat'},
             8: {'train': 'cow', 'test': 'dog'},
             9: {'train': 'keyboard', 'test': 'dog'},
}

# Initialize test accuracy dictionaries
unedited = {scenario: {"pcbm": [], "pcbm-h": []} for scenario in range(10)}
prune = {scenario: {"pcbm": [], "pcbm-h": []} for scenario in range(10)}
prune_normalize = {scenario: {"pcbm": [], "pcbm-h": []} for scenario in range(10)}
fine_tune = {scenario: {"pcbm": [], "pcbm-h": []} for scenario in range(10)}

concept_bank_path = os.path.join(CONCEPT_BANK_DIR, f"{CONCEPT}_{BACKBONE}_{SEED}_{N_SAMPLES}.pkl")

# Check if the concept bank file already exists
if not os.path.exists(concept_bank_path):
    get_concepts_dataset(
        backbone_name=BACKBONE, 
        dataset_name=CONCEPT, 
        out_dir=CONCEPT_BANK_DIR, 
        device=DEVICE, 
        seed=SEED, 
        num_workers=NUM_WORKERS, 
        batch_size=BATCH_SIZE, 
        C=C_VALUES, 
        n_samples=N_SAMPLES
    )

for scenario in range(10):
    model_path = os.path.join(PCBM_MODELS_DIR, f"pcbm_{DATASET + '_' + str(scenario)}__{BACKBONE}__{CONCEPT}__lam-{LAM_VALUE}__alpha-{ALPHA}__seed-{SEED}.ckpt")

    run_info_pcbm = get_pcbm(
        baseline=False,
        validation=False, 
        concept_bank=concept_bank_path, 
        out_dir=PCBM_MODELS_DIR, 
        dataset=DATASET + "_" + str(scenario), 
        backbone_name=BACKBONE, 
        device=DEVICE, 
        seed=SEED, 
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
        seed=SEED,
        num_workers=NUM_WORKERS,
        num_epochs=10,
        lr=0.01,
        l2_penalty=0.01
    )
    unedited[scenario]["pcbm-h"].append(run_info_pcbm_h["test_acc"])

"""
for scenario, config in SCENARIOS.items():
    hybrid_model_path = os.path.join(PCBM_H_MODELS_DIR, f"pcbm-hybrid_{DATASET + '_' + str(scenario)}__{BACKBONE}__{CONCEPT}__lr-{0.01}__l2_penalty-{0.01}__seed-{SEED}.ckpt")
    posthoc_h_layer = torch.load(hybrid_model_path)
    posthoc_h_layer = posthoc_h_layer.eval()

    hybrid_model = PosthocHybridCBM(posthoc_h_layer)
    hybrid_model = hybrid_model.to(DEVICE)

    train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls = load_or_compute_projections(BACKBONE, posthoc_h_layer, train_loader, test_loader, **kwargs)

    norms = []
    for idx, cls in hybrid_model.idx_to_class.items():
        cls_weights = hybrid_model.classifier.weight[idx]
        print(cls, cls_weights)
        for j in cls_weights:
            print(hybrid_model.names[j])

        norms.append(torch.norm(cls_weights))

        # Prune
        if cls == config['class']:
            prune_idx = hybrid_model.names.index(config['prune'])
            hybrid_model.classifier.weight[idx][prune_idx] = 0

    run_info_pcbm = evaluate_model(hybrid_model, test_loader, num_classes, **kwargs)
    run_info_pcbm_h = evaluate_model(posthoc_h_layer, test_loader, num_classes, **kwargs)

    prune[scenario]["pcbm"].append(run_info_pcbm["test_acc"])
    prune[scenario]["pcbm-h"].append(run_info_pcbm_h["test_acc"])

    # Prune + normalize
    for idx, cls in hybrid_model.idx_to_class.items():
        hybrid_model.classifier.weight[idx] = hybrid_model.classifier.weight[idx] / norms[-1]
        
    run_info_pcbm = evaluate_model(hybrid_model, test_loader, num_classes, **kwargs)
    run_info_pcbm_h = evaluate_model(posthoc_h_layer, test_loader, num_classes, **kwargs)

    prune_normalize[scenario]["pcbm"].append(run_info_pcbm["test_acc"])
    prune_normalize[scenario]["pcbm-h"].append(run_info_pcbm_h["test_acc"])
"""
# Fine-tune

write_statistics(unedited, "test_accuracy_unedited.json")
#write_statistics(prune, "test_accuracy_prune.json")
#write_statistics(prune_normalize, "test_accuracy_prune_normalize.json")
#write_statistics(fine_tune, "test_accuracy_fine_tune.json")

    
