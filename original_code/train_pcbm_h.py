import argparse
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import sys
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from .data import get_dataset
from .concepts import ConceptBank
from .models import PosthocLinearCBM, PosthocHybridCBM, get_model
from .training_tools import load_or_compute_projections, AverageMeter, MetricComputer


@torch.no_grad()
def eval_model(posthoc_layer, loader, num_classes, **kwargs):
    epoch_summary = {"Accuracy": AverageMeter()}
    tqdm_loader = tqdm(loader)
    computer = MetricComputer(n_classes=num_classes)
    all_preds = []
    all_labels = []
    
    for batch_X, batch_Y in tqdm(loader):
        batch_X, batch_Y = batch_X.to(kwargs['device']), batch_Y.to(kwargs['device']) 
        out = posthoc_layer(batch_X)            
        all_preds.append(out.detach().cpu().numpy())
        all_labels.append(batch_Y.detach().cpu().numpy())
        metrics = computer(out, batch_Y) 
        epoch_summary["Accuracy"].update(metrics["accuracy"], batch_X.shape[0]) 
        summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
        summary_text = "Eval - " + " ".join(summary_text)
        tqdm_loader.set_description(summary_text)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if all_labels.max() == 1:
        auc = roc_auc_score(all_labels, softmax(all_preds, axis=1)[:, 1])
        return auc
    return epoch_summary["Accuracy"].avg


def train_hybrid(train_loader, val_loader, posthoc_layer, optimizer, num_classes, **kwargs):
    cls_criterion = nn.CrossEntropyLoss()
    for epoch in range(1, kwargs['num_epochs']+1):
        print(f"Epoch: {epoch}")
        epoch_summary = {"CELoss": AverageMeter(),
                         "Accuracy": AverageMeter()}
        tqdm_loader = tqdm(train_loader)
        computer = MetricComputer(n_classes=num_classes)
        for batch_X, batch_Y in tqdm(train_loader):
            batch_X, batch_Y = batch_X.to(kwargs['device']), batch_Y.to(kwargs['device'])
            optimizer.zero_grad()
            out, projections = posthoc_layer(batch_X, return_dist=True)
            cls_loss = cls_criterion(out, batch_Y)
            loss = cls_loss + kwargs['l2_penalty']*(posthoc_layer.residual_classifier.weight**2).mean()
            loss.backward()
            optimizer.step()
            
            epoch_summary["CELoss"].update(cls_loss.detach().item(), batch_X.shape[0])
            metrics = computer(out, batch_Y) 
            epoch_summary["Accuracy"].update(metrics["accuracy"], batch_X.shape[0])

            summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
            summary_text = " ".join(summary_text)
            tqdm_loader.set_description(summary_text)
        
        latest_info = dict()
        latest_info["epoch"] = epoch
        latest_info["args"] = kwargs
        latest_info["train_acc"] = epoch_summary["Accuracy"]
        latest_info["test_acc"] = eval_model(posthoc_layer, val_loader, num_classes, **kwargs)
        print("Final test acc: ", latest_info["test_acc"])
    return latest_info


def get_pcbm_h(**kwargs):
    # Load the PCBM
    posthoc_layer = torch.load(kwargs['pcbm_path'])
    posthoc_layer = posthoc_layer.eval()
    kwargs['backbone_name'] = posthoc_layer.backbone_name
    backbone, preprocess = get_model(**kwargs)
    backbone = backbone.to(kwargs['device'])
    backbone.eval()

    train_loader, test_loader, idx_to_class, classes = get_dataset(preprocess, **kwargs)
    num_classes = len(classes)
    
    # We use the precomputed embeddings and projections.
    train_embs, _, train_lbls, test_embs, _, test_lbls = load_or_compute_projections(backbone, posthoc_layer, train_loader, test_loader, **kwargs)

    train_loader = DataLoader(TensorDataset(torch.tensor(train_embs).float(), torch.tensor(train_lbls).long()), batch_size=kwargs['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_embs).float(), torch.tensor(test_lbls).long()), batch_size=kwargs['batch_size'], shuffle=False)

    # Initialize PCBM-h
    hybrid_model = PosthocHybridCBM(posthoc_layer)
    hybrid_model = hybrid_model.to(kwargs['device'])
    
    # Initialize the optimizer
    hybrid_optimizer = torch.optim.Adam(hybrid_model.residual_classifier.parameters(), lr=kwargs['lr'])
    hybrid_model.residual_classifier = hybrid_model.residual_classifier.float()
    hybrid_model.bottleneck = hybrid_model.bottleneck.float()
    
    # Train PCBM-h
    run_info = train_hybrid(train_loader, test_loader, hybrid_model, hybrid_optimizer, num_classes, **kwargs)
    
    conceptbank_source = kwargs['concept_bank'].split("/")[-1].split(".")[0]
    hybrid_model_path = os.path.join(kwargs['out_dir'],
                              f"pcbm-hybrid_{kwargs['dataset']}__{kwargs['backbone_name']}__{conceptbank_source}__lr-{kwargs['lr']}__l2_penalty-{kwargs['l2_penalty']}__seed-{kwargs['seed']}.ckpt")
    torch.save(hybrid_model, hybrid_model_path)

    run_info_file = os.path.join(kwargs['out_dir'],
                            f"run_info-pcbm-hybrid_{kwargs['dataset']}__{kwargs['backbone_name']}__{conceptbank_source}__lr-{kwargs['lr']}__l2_penalty-{kwargs['l2_penalty']}__seed-{kwargs['seed']}.pkl")
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)
    
    print(f"Saved to {hybrid_model_path}, {run_info_file}")

    return run_info
