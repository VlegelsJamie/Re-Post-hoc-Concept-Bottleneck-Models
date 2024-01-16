from original_code import get_concepts_dataset

# Set your parameters
backbone_name = "clip:RN50"
dataset_name = "broden"
out_dir = "../concept_banks/"
device = "cuda"  # or "cpu" if you don't have a GPU
seed = 1
num_workers = 4
batch_size = 100
C = [0.1]
n_samples = 50

# Run the learning process
get_concepts_dataset(backbone_name=backbone_name, dataset_name=dataset_name, out_dir=out_dir, device=device, seed=seed, 
                       num_workers=num_workers, batch_size=batch_size, C=C, n_samples=n_samples)

"""
!python learn_concepts_dataset.py --dataset-name="broden" --backbone-name="clip:RN50" --C 0.1 --n-samples=50 --out-dir=$OUTPUT_DIR_CONCEPTS

# Learning CUB Concepts with ResNet18 Backbone
# !python3 learn_concepts_dataset.py --dataset-name="cub" --backbone-name="resnet18_cub" --C 0.1 --n-samples=50 --out-dir=$OUTPUT_DIR_CONCEPTS --device="cpu"

# Learning Derm7pt Concepts with Inception Backbone
# !python3 learn_concepts_dataset.py --dataset-name="derm7pt" --backbone-name="ham10000_inception" --C 0.1 --n-samples=50 --out-dir=$OUTPUT_DIR_CONCEPTS --device="cpu"


# Training PCBMs
OUTPUT_DIR_PCBM = "/PCMBs/"

# Learning CIFAR10 PCBM with Broden Concepts
#!python train_pcbm.py --concept-bank="${OUTPUT_DIR_CONCEPTS}/.pkl" --dataset="cifar10" --backbone-name="clip:RN50" --out-dir=$OUTPUT_DIR --lam=2.0

# Learning CIFAR100 PCBM with Broden Concepts
#!python train_pcbm.py --concept-bank="${OUTPUT_DIR_CONCEPTS}/.pkl" --dataset="cifar100" --backbone-name="clip:RN50" --out-dir=$OUTPUT_DIR --lam=2.0

# Learning COCO-Stuff PCBM with Broden Concepts
#!python train_pcbm.py --concept-bank="${OUTPUT_DIR_CONCEPTS}/.pkl" --dataset="" --backbone-name="clip:RN50" --out-dir=$OUTPUT_DIR --lam=0.001

# Learning CUB PCBM with CUB Concepts
#!python train_pcbm.py --concept-bank="${OUTPUT_DIR_CONCEPTS}/cub_resnet18_cub_0.1_50.pkl" --dataset="cub" --backbone-name="resnet18_cub" --out-dir=$OUTPUT_DIR --lam=0.01

# Learning HAM10k PCBM with Derm7pt Concepts
#!python train_pcbm.py --concept-bank="${OUTPUT_DIR_CONCEPTS}/.pkl" --dataset="ham10000" --backbone-name="ham10000_inception" --out-dir=$OUTPUT_DIR --lam=2.0

# Learning SIIM-ISIC PCBM with Derm7pt Concepts
#!python train_pcbm.py --concept-bank="${OUTPUT_DIR_CONCEPTS}/.pkl" --dataset="" --backbone-name="ham10000_inception" --out-dir=$OUTPUT_DIR --lam=0.001


# Training PCBM-hs
#pcbm_path="/path/to/pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"
#python3 train_pcbm_h.py --concept-bank="${OUTPUT_DIR}/cub_resnet18_cub_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=$OUTPUT_DIR --dataset="cub"
"""