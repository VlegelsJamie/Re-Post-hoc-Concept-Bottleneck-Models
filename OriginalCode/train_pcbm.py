import argparse
import os
import pickle
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score


from .data import get_dataset
from .concepts import ConceptBank
from .models import PosthocLinearCBM, get_model
from .training_tools import load_or_compute_projections


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    return parser.parse_args()


def run_linear_probe(train_data, test_data, **kwargs):
    train_features, train_labels = train_data
    test_features, test_labels = test_data
    
    # We converged to using SGDClassifier. 
    # It's fine to use other modules here, this seemed like the most pedagogical option.
    # We experimented with torch modules etc., and results are mostly parallel.
    classifier = SGDClassifier(random_state=kwargs['seed'], loss="log_loss",
                               alpha=kwargs['lam'], l1_ratio=kwargs['alpha'], verbose=0,
                               penalty="elasticnet", max_iter=10000)
    classifier.fit(train_features, train_labels)

    train_predictions = classifier.predict(train_features)
    train_accuracy = np.mean((train_labels == train_predictions).astype(float)) * 100.
    predictions = classifier.predict(test_features)
    test_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.

    # Compute class-level accuracies. Can later be used to understand what classes are lacking some concepts.
    cls_acc = {"train": {}, "test": {}}
    for lbl in np.unique(train_labels):
        test_lbl_mask = test_labels == lbl
        train_lbl_mask = train_labels == lbl
        cls_acc["test"][lbl] = np.mean((test_labels[test_lbl_mask] == predictions[test_lbl_mask]).astype(float))
        cls_acc["train"][lbl] = np.mean(
            (train_labels[train_lbl_mask] == train_predictions[train_lbl_mask]).astype(float))
        print(f"{lbl}: {cls_acc['test'][lbl]}")

    run_info = {"train_acc": train_accuracy, "test_acc": test_accuracy,
                "cls_acc": cls_acc,
                }

    # If it's a binary task, we compute auc
    if test_labels.max() == 1:
        run_info["test_auc"] = roc_auc_score(test_labels, classifier.decision_function(test_features))
        run_info["train_auc"] = roc_auc_score(train_labels, classifier.decision_function(train_features))
    return run_info, classifier.coef_, classifier.intercept_


def get_pcbm(**kwargs):
    all_concepts = pickle.load(open(kwargs['concept_bank'], 'rb'))
    all_concept_names = list(all_concepts.keys())
    print(f"Bank path: {kwargs['concept_bank']}. {len(all_concept_names)} concepts will be used.")
    concept_bank = ConceptBank(all_concepts, kwargs['device'])

    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(backbone_name=kwargs['backbone_name'], **kwargs)
    backbone = backbone.to(kwargs['device'])
    backbone.eval()

    train_loader, test_loader, idx_to_class, classes = get_dataset(preprocess, **kwargs)
    
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    # which means a bank learned with 100 samples per concept with C=0.1 regularization parameter for the SVM. 
    # See `learn_concepts_dataset.py` for details.
    conceptbank_source = kwargs['concept_bank'].split("/")[-1].split(".")[0] 
    num_classes = len(classes)
    
    # Initialize the PCBM module.
    posthoc_layer = PosthocLinearCBM(concept_bank, backbone_name=kwargs['backbone_name'], idx_to_class=idx_to_class, n_classes=num_classes)
    posthoc_layer = posthoc_layer.to(kwargs['device'])

    # We compute the projections and save to the output directory. This is to save time in tuning hparams / analyzing projections.
    train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls = load_or_compute_projections(backbone, posthoc_layer, train_loader, test_loader, **kwargs)
    
    run_info, weights, bias = run_linear_probe((train_projs, train_lbls), (test_projs, test_lbls), **kwargs)
    
    # Convert from the SGDClassifier module to PCBM module.
    posthoc_layer.set_weights(weights=weights, bias=bias)

    # Sorry for the model path hack. Probably i'll change this later.
    model_path = os.path.join(kwargs['out_dir'],
                              f"pcbm_{kwargs['dataset']}__{kwargs['backbone_name']}__{conceptbank_source}__lam:{kwargs['lam']}__alpha:{kwargs['alpha']}__seed:{kwargs['seed']}.ckpt")
    torch.save(posthoc_layer, model_path)

    # Again, a sad hack.. Open to suggestions
    run_info_file = model_path.replace("pcbm", "run_info-pcbm")
    run_info_file = run_info_file.replace(".ckpt", ".pkl")
    run_info_file = os.path.join(kwargs['out_dir'], run_info_file)
    
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)

    
    if num_classes > 1:
        # Prints the Top-5 Concept Weigths for each class.
        print(posthoc_layer.analyze_classifier(k=5))

    print(f"Model saved to : {model_path}")
    print(run_info)


def main():
    args = config()
    get_pcbm(concept_bank=args.concept_bank, out_dir=args.out_dir, dataset=args.dataset, backbone_name=args.backbone_name, device=args.device, seed=args.seed, 
             batch_size=args.batch_size, num_workers=args.num_workers, alpha=args.alpha, lam=args.lam, lr=args.lr)

if __name__ == "__main__":
    main()
