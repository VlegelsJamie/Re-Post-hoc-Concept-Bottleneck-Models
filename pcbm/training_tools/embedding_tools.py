import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


def unpack_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1]
    elif len(batch) == 2:
        return batch
    else:
        raise ValueError()


@torch.no_grad()
def get_projections(backbone, posthoc_layer, loader, **kwargs):
    all_projs, all_embs, all_lbls = None, None, None
    for batch in tqdm(loader):
        batch_X, batch_Y = unpack_batch(batch)
        batch_X = batch_X.to(kwargs['device'])
        if "clip" in kwargs['backbone_name']:
            embeddings = backbone.encode_image(batch_X).detach().float()
        else:
            embeddings = backbone(batch_X).detach()
        projs = posthoc_layer.compute_dist(embeddings).detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()
        if all_embs is None:
            all_embs = embeddings
            all_projs = projs
            all_lbls = batch_Y.numpy()
        else:
            all_embs = np.concatenate([all_embs, embeddings], axis=0)
            all_projs = np.concatenate([all_projs, projs], axis=0)
            all_lbls = np.concatenate([all_lbls, batch_Y.numpy()], axis=0)
    return all_embs, all_projs, all_lbls


class EmbDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y
    def __len__(self):
        return len(self.data)


def load_or_compute_projections(backbone, posthoc_layer, train_loader, test_loader, **kwargs):
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    conceptbank_source = kwargs['concept_bank'].split("/")[-1].split("_")[0] 
    
    # To make it easier to analyize results/rerun with different params, we'll extract the embeddings and save them
    train_file = f"train-embs_{kwargs['dataset']}__{kwargs['backbone_name']}__{conceptbank_source}.npy"
    test_file = f"test-embs_{kwargs['dataset']}__{kwargs['backbone_name']}__{conceptbank_source}.npy"
    train_proj_file = f"train-proj_{kwargs['dataset']}__{kwargs['backbone_name']}__{conceptbank_source}.npy"
    test_proj_file = f"test-proj_{kwargs['dataset']}__{kwargs['backbone_name']}__{conceptbank_source}.npy"
    train_lbls_file = f"train-lbls_{kwargs['dataset']}__{kwargs['backbone_name']}__{conceptbank_source}_lbls.npy"
    test_lbls_file = f"test-lbls_{kwargs['dataset']}__{kwargs['backbone_name']}__{conceptbank_source}_lbls.npy"
    
    out_dir = kwargs['out_dir'].split("/")[0]
    train_file = os.path.join(out_dir, train_file)
    test_file = os.path.join(out_dir, test_file)
    train_proj_file = os.path.join(out_dir, train_proj_file)
    test_proj_file = os.path.join(out_dir, test_proj_file)
    train_lbls_file = os.path.join(out_dir, train_lbls_file)
    test_lbls_file = os.path.join(out_dir, test_lbls_file)

    if os.path.exists(train_proj_file):
        train_embs = np.load(train_file)
        test_embs = np.load(test_file)
        train_projs = np.load(train_proj_file)
        test_projs = np.load(test_proj_file)
        train_lbls = np.load(train_lbls_file)
        test_lbls = np.load(test_lbls_file)

    else:
        train_embs, train_projs, train_lbls = get_projections(backbone, posthoc_layer, train_loader, **kwargs)
        test_embs, test_projs, test_lbls = get_projections(backbone, posthoc_layer, test_loader, **kwargs)

        np.save(train_file, train_embs)
        np.save(test_file, test_embs)
        np.save(train_proj_file, train_projs)
        np.save(test_proj_file, test_projs)
        np.save(train_lbls_file, train_lbls)
        np.save(test_lbls_file, test_lbls)
    
    return train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls

