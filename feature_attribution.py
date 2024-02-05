import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from torchvision import datasets, transforms

from pcbm.data import get_dataset
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, get_model
from pcbm.training_tools import load_or_compute_projections


def plot_integrated_gradients(integrated_grads, original_img, title=None):
    # Convert tensor to numpy for visualization
    if torch.is_tensor(integrated_grads):
        integrated_grads = integrated_grads.squeeze(0).cpu().numpy()
    
    # Aggregate gradients across color channels
    aggregated_grads = np.sum(np.abs(integrated_grads), axis=0)
    
    # Normalize the aggregated gradients to [0, 1] range for visualization
    aggregated_grads = (aggregated_grads - np.min(aggregated_grads)) / (np.max(aggregated_grads) - np.min(aggregated_grads) + 1e-8)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display original image
    axs[0].imshow(original_img)
    axs[0].axis('off')
    axs[0].set_title('Original Image')
    
    # Display the saliency map
    axs[1].imshow(aggregated_grads, cmap='hot', interpolation='nearest')
    axs[1].axis('off')
    axs[1].set_title('Saliency Map')
    
    if title:
        plt.suptitle(title)

    plt.show()


def integrated_gradients(input_img, posthoc_layer, backbone, concept_vector_idx, baseline=None, steps=50):
    # If no baseline is provided, use a zero image
    if baseline is None:
        baseline = torch.zeros_like(input_img)
    
    # Generate interpolated images between baseline and input
    interpolated_images = [baseline + (float(i) / steps) * (input_img - baseline) for i in range(0, steps + 1)]

    # Compute gradients for each interpolated image
    gradients = []
    for img in interpolated_images:
        img.requires_grad_(True)
        img.retain_grad()

        embeddings = backbone.encode_image(img) 
        embeddings.requires_grad_(True)
        embeddings.retain_grad()

        concept_score = posthoc_layer.compute_dist(embeddings)[0, concept_vector_idx] 
        concept_score.backward()

        gradients.append(img.grad.detach().clone())

        img.grad.zero_()
        embeddings.grad.zero_()

    # Average the gradients across all steps
    avg_gradients = torch.mean(torch.stack(gradients), dim=0)
    
    # Compute integrated gradients
    integrated_grads = (input_img - baseline) * avg_gradients

    return integrated_grads

# Constants
DEVICE = "cpu"
NUM_WORKERS = 4
BATCH_SIZE = 64
OUT_DIR = "trained_models/pcbm_models/experiment_featureattribution/"

# Load the PCBM
pcbm_bank_path = "trained_models/pcbm_models/experiment_conceptdataset/pcbm_cifar10__clip-RN50__broden__lam-2.0__alpha-0.99__seed-42.ckpt"
pcbm_clip_path = "trained_models/pcbm_models/experiment_multimodal/pcbm_cifar10__clip-RN50__multimodal__lam-2__alpha-0.99__seed-42.ckpt"

pcbm_bank = torch.load(pcbm_bank_path)
pcbm_bank = pcbm_bank.eval()

pcbm_clip = torch.load(pcbm_clip_path)
pcbm_clip = pcbm_clip.eval()

print(pcbm_bank.analyze_classifier(k=5))
print(pcbm_clip.analyze_classifier(k=5))

backbone_name = pcbm_bank.backbone_name
backbone, preprocess = get_model(backbone_name=backbone_name, full_model=False, device=DEVICE, out_dir=OUT_DIR)
backbone = backbone.to(DEVICE)
backbone.eval()

print(pcbm_bank.names)
index = pcbm_clip.names.index("windshield")
print(index)

testset = datasets.CIFAR10(root=OUT_DIR, train=False, download=True, transform=preprocess)
testset_unnormalized = datasets.CIFAR10(root=OUT_DIR, train=False, download=True, transform=None)

classes = testset.classes
class_to_idx = {c: i for (i, c) in enumerate(classes)}

# Get the index for the 'automobile' class
automobile_idx = class_to_idx['automobile']

# Filter the dataset for 'automobile' images
automobile_images = [(img, label) for img, label in testset if label == automobile_idx]
automobile_images_unnormalized = [(img, label) for img, label in testset_unnormalized if label == automobile_idx]

n_px = 224

transformation = transforms.Compose([
    transforms.Resize((n_px, n_px), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(n_px)
])

for i in range(10):
    input_img, label = automobile_images[i]
    input_img_unnormalized, label = automobile_images_unnormalized[i]
    input_img_unnormalized = transformation(input_img_unnormalized)

    npimg = np.asarray(input_img_unnormalized)

    integrated_grads = integrated_gradients(input_img.unsqueeze(0).to(DEVICE), pcbm_clip, backbone, index)
    plot_integrated_gradients(integrated_grads, npimg, title='Integrated Gradients Visualization')
