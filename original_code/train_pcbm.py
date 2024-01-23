import argparse
import os
import pickle
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


from .data import get_dataset
from .concepts import ConceptBank
from .models import PosthocLinearCBM, get_model
from .training_tools import load_or_compute_projections


@torch.no_grad()
def evaluate_backbone(model, train_loader, test_loader, device):
    train_predictions, train_labels = [], []
    test_predictions, test_labels = [], []

    # Evaluate on train data
    for batch in train_loader:
        batch_X, batch_Y = batch
        batch_X = batch_X.to(device)
        outputs = model(batch_X).float()
        train_predictions.extend(outputs.cpu().numpy())
        train_labels.extend(batch_Y.numpy())

    # Evaluate on test data
    for batch in test_loader:
        batch_X, batch_Y = batch
        batch_X = batch_X.to(device)
        outputs = model(batch_X).float()
        test_predictions.extend(outputs.cpu().numpy())
        test_labels.extend(batch_Y.numpy())

    # Convert to numpy arrays
    train_predictions = np.array(train_predictions)
    train_labels = np.array(train_labels)
    test_predictions = np.array(test_predictions)
    test_labels = np.array(test_labels)

    # Calculate accuracies
    train_accuracy = np.mean((train_labels == np.argmax(train_predictions, axis=1)).astype(float)) * 100.
    test_accuracy = np.mean((test_labels == np.argmax(test_predictions, axis=1)).astype(float)) * 100.

    # Compute class-level accuracies
    cls_acc = {"train": {}, "test": {}}
    for lbl in np.unique(train_labels):
        train_lbl_mask = train_labels == lbl
        test_lbl_mask = test_labels == lbl
        cls_acc["train"][lbl] = np.mean((train_labels[train_lbl_mask] == np.argmax(train_predictions[train_lbl_mask], axis=1)).astype(float))
        cls_acc["test"][lbl] = np.mean((test_labels[test_lbl_mask] == np.argmax(test_predictions[test_lbl_mask], axis=1)).astype(float))

    run_info = {"train_acc": train_accuracy, "test_acc": test_accuracy, "cls_acc": cls_acc}

    # Compute AUC for binary tasks
    if np.unique(test_labels).size == 2:
        run_info["train_acc"] = roc_auc_score(train_labels, train_predictions[:, 1])
        run_info["test_acc"] = roc_auc_score(test_labels, test_predictions[:, 1])

    return run_info


def run_linear_probe(train_data, test_data, norm, **kwargs):
    train_features, train_labels = train_data
    test_features, test_labels = test_data
    
    # We converged to using SGDClassifier. 
    # It's fine to use other modules here, this seemed like the most pedagogical option.
    # We experimented with torch modules etc., and results are mostly parallel.
    classifier = SGDClassifier(random_state=kwargs['seed'], loss="log_loss",
                               alpha=kwargs['lam'] / norm, l1_ratio=kwargs['alpha'], verbose=0,
                               penalty="elasticnet", max_iter=5000)
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
        run_info["test_acc"] = roc_auc_score(test_labels, classifier.decision_function(test_features))
        run_info["train_acc"] = roc_auc_score(train_labels, classifier.decision_function(train_features))
    return run_info, classifier


def get_pcbm(**kwargs):
    all_concepts = pickle.load(open(kwargs['concept_bank'], 'rb'))
    all_concept_names = list(all_concepts.keys())
    num_concepts = len(all_concept_names)
    print(f"Bank path: {kwargs['concept_bank']}. {num_concepts} concepts will be used.")
    concept_bank = ConceptBank(all_concepts, kwargs['device'])

    # Get the backbone from the model zoo.
    if kwargs['baseline']:
        model, backbone, preprocess = get_model(full_model=True, **kwargs)
        if model is not None:
            model = model.to(kwargs['device'])
            model.eval()
    else:
        backbone, preprocess = get_model(**kwargs)
    backbone = backbone.to(kwargs['device'])
    backbone.eval()

    train_loader, test_loader, idx_to_class, classes = get_dataset(preprocess, **kwargs)
    
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    # which means a bank learned with 100 samples per concept with C=0.1 regularization parameter for the SVM. 
    # See `learn_concepts_dataset.py` for details.
    conceptbank_source = kwargs['concept_bank'].split("/")[-1].split("_")[0] 
    num_classes = len(classes)
    
    # Initialize the PCBM module.
    posthoc_layer = PosthocLinearCBM(concept_bank, backbone_name=kwargs['backbone_name'], idx_to_class=idx_to_class, n_classes=num_classes)
    posthoc_layer = posthoc_layer.to(kwargs['device'])

    # We compute the projections and save to the output directory. This is to save time in tuning hparams / analyzing projections.
    train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls = load_or_compute_projections(backbone, posthoc_layer, train_loader, test_loader, **kwargs)

    if kwargs['validation']:
        os.makedirs(kwargs['validation'], exist_ok=True)
        norm = num_classes
        train_embs, test_embs, train_projs, test_projs, train_lbls, test_lbls = train_test_split(
            train_embs, train_projs, train_lbls, test_size=0.2, random_state=kwargs['seed'])
        
        if kwargs['baseline']:
            model_path_baseline = os.path.join(kwargs['validation'],
                                    f"validation_baseline_{kwargs['dataset']}__{kwargs['backbone_name']}__lam-{kwargs['lam']}__alpha-{kwargs['alpha']}__seed-{kwargs['seed']}.ckpt")
            run_info_file_baseline = os.path.join(kwargs['validation'],
                                f"run_info-validation_baseline_{kwargs['dataset']}__{kwargs['backbone_name']}__lam-{kwargs['lam']}__alpha-{kwargs['alpha']}__seed-{kwargs['seed']}.pkl")
        
        model_path = os.path.join(kwargs['validation'],
                                f"validation_{kwargs['dataset']}__{kwargs['backbone_name']}__lam-{kwargs['lam']}__alpha-{kwargs['alpha']}__seed-{kwargs['seed']}.ckpt")
        run_info_file = os.path.join(kwargs['validation'],
                              f"run_info-validation_{kwargs['dataset']}__{kwargs['backbone_name']}__lam-{kwargs['lam']}__alpha-{kwargs['alpha']}__seed-{kwargs['seed']}.pkl")
    else:
        norm = num_concepts * num_classes

        if kwargs['baseline']:
            model_path_baseline = os.path.join(kwargs['baseline'],
                                    f"baseline_{kwargs['dataset']}__{kwargs['backbone_name']}__lam-{kwargs['lam']}__alpha-{kwargs['alpha']}__seed-{kwargs['seed']}.ckpt")
            run_info_file_baseline = os.path.join(kwargs['baseline'],
                                f"run_info-baseline_{kwargs['dataset']}__{kwargs['backbone_name']}__lam-{kwargs['lam']}__alpha-{kwargs['alpha']}__seed-{kwargs['seed']}.pkl")
        
        model_path = os.path.join(kwargs['out_dir'],
                              f"pcbm_{kwargs['dataset']}__{kwargs['backbone_name']}__{conceptbank_source}__lam-{kwargs['lam']}__alpha-{kwargs['alpha']}__seed-{kwargs['seed']}.ckpt")
        run_info_file = os.path.join(kwargs['out_dir'],
                              f"run_info-pcbm_{kwargs['dataset']}__{kwargs['backbone_name']}__{conceptbank_source}__lam-{kwargs['lam']}__alpha-{kwargs['alpha']}__seed-{kwargs['seed']}.pkl")

    # Compute baseline by training a linear probe on the embeddings of backbone model.
    if kwargs['baseline']:
        os.makedirs(kwargs['baseline'], exist_ok=True)

        if kwargs["backbone_name"] == "resnet18_cub" or kwargs["backbone_name"] == "ham10000_inception":
            run_info_baseline = evaluate_backbone(model, train_loader, test_loader, kwargs['device'])
        else:
            run_info_baseline, classifier_baseline = run_linear_probe((train_embs, train_lbls), (test_embs, test_lbls), norm, **kwargs)

            with open(model_path_baseline, "wb") as f:
                pickle.dump(classifier_baseline, f)

            print(f"Baseline model saved to : {model_path_baseline}")

        with open(run_info_file_baseline, "wb") as f:
            pickle.dump(run_info_baseline, f)

        print(run_info_baseline)
    
    run_info_pcbm, classifier_pcbm = run_linear_probe((train_projs, train_lbls), (test_projs, test_lbls), norm, **kwargs)
    
    # Convert from the SGDClassifier module to PCBM module.
    posthoc_layer.set_weights(weights=classifier_pcbm.coef_, bias=classifier_pcbm.intercept_)

    torch.save(posthoc_layer, model_path)
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info_pcbm, f)

    if num_classes > 1:
        # Prints the Top-5 Concept Weigths for each class.
        print(posthoc_layer.analyze_classifier(k=5))

    print(f"Model saved to : {model_path}")
    print(run_info_pcbm)

    if kwargs['baseline']:
        return run_info_pcbm, run_info_baseline
    else:
        return run_info_pcbm
