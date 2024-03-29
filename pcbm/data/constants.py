import os

# CUB Constants
# CUB data is downloaded from the CBM release.
# Dataset: https://worksheets.codalab.org/rest/bundles/0xd013a7ba2e88481bbc07e787f73109f5/ 
# Metadata and splits: https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683
CUB_DATA_DIR = "datasets/cub_dataset/"
CUB_PROCESSED_DIR = "datasets/cub_processed/"


# Derm data constants
# Derm7pt is obtained from : https://derm.cs.sfu.ca/Welcome.html
DERM7_FOLDER = "datasets/derm7pt_concepts/"
DERM7_META = os.path.join(DERM7_FOLDER, "meta", "meta.csv")
DERM7_TRAIN_IDX = os.path.join(DERM7_FOLDER, "meta", "train_indexes.csv")
DERM7_VAL_IDX = os.path.join(DERM7_FOLDER, "meta", "valid_indexes.csv")


# BRODEN concept bank
BRODEN_CONCEPTS = "datasets/broden_concepts/"


# Ham10000 can be obtained from : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
HAM10K_DATA_DIR = "datasets/ham10k_dataset/"


# COCO-Stuff dataset
COCO_DATA_DIR = "datasets/coco_dataset/"
COCO_IMAGES = os.path.join(COCO_DATA_DIR, "images")
COCO_META = os.path.join(COCO_DATA_DIR, "meta.json")
COCO_LABELS = os.path.join(COCO_DATA_DIR, "labels.txt")
COCO_ANNOTATIONS = os.path.join(COCO_DATA_DIR, "annotations")


# SIIM-ISIC dataset
ISIC_DATA_DIR = "datasets/isic_dataset/"


# Metashift dataset
METASHIFT_DATA_DIR = "datasets/metashift_dataset/"