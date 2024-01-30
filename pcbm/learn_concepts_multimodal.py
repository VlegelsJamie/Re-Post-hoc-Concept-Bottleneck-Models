import requests
import os
import pickle
import torch
import clip
import argparse
import numpy as np
from tqdm import tqdm


concept_cache = {}


def get_single_concept_data(cls_name):
    if cls_name in concept_cache:
        return concept_cache[cls_name]
    
    all_concepts = []
    
    # Has relations
    has_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/HasA&start=/c/en/{}"
    obj = requests.get(has_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['end']['label'])
    
    # Made of relations
    madeof_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/MadeOf&start=/c/en/{}"
    obj = requests.get(madeof_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['end']['label'])
    
    # Properties of things
    property_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/HasProperty&start=/c/en/{}"
    obj = requests.get(property_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['end']['label'])
    
    # Categorization concepts
    is_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/IsA&start=/c/en/{}"
    obj = requests.get(is_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        if edge["weight"] <= 1:
            continue
        all_concepts.append(edge['end']['label'])
    
    # Parts of things
    parts_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/PartOf&end=/c/en/{}"
    obj = requests.get(parts_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['start']['label'])
    
    all_concepts = [c.lower() for c in all_concepts]
    # Drop the "a " for concepts defined like "a {concept}".
    all_concepts = [c.replace("a ", "") for c in all_concepts]
    # Drop all empty concepts.
    all_concepts = [c for c in all_concepts if c!=""]
    # Make each concept unique in the set.
    all_concepts = set(all_concepts)
    
    concept_cache[cls_name] = all_concepts
    
    return all_concepts


def get_concept_data(all_classes):
    all_concepts = set()
    # Collect concepts that are relevant to each class
    for cls_name in all_classes:
        print(f"Pulling concepts for {cls_name}")
        all_concepts |= get_single_concept_data(cls_name)
    return all_concepts


def clean_concepts(scenario_concepts, out_dir):
    """
    Clean the plurals, trailing whitespaces etc.
    """
    from nltk.stem.wordnet import WordNetLemmatizer
    import nltk

    # We use nltk to handle plurals, multiples of the same words etc.
    nltk.data.path.append(out_dir)
    nltk.download("wordnet", download_dir=out_dir)
    nltk.download("omw-1.4", download_dir=out_dir)
    Lem = WordNetLemmatizer()

    scenario_concepts_rec = []
    for c_prev in scenario_concepts:
        c = c_prev
        c = c.strip()
        c_subwords = c.split(" ")
        # If a concept is made of more than 2 words, we drop it.
        if len(c_subwords) > 2:
            print("skipping long concept", c_prev)
            continue
        # Lemmatize words to help eliminate non-unique concepts etc.
        for i, csw in enumerate(c_subwords):
            c_subwords[i] = Lem.lemmatize(csw)
        lemword = " ".join(c_subwords)
        if c_prev == lemword:
            scenario_concepts_rec.append(c)
        else:
            if lemword in scenario_concepts:
                print(c, lemword)
            else:
                scenario_concepts_rec.append(c)
    scenario_concepts_rec = list(set(scenario_concepts_rec))
    return scenario_concepts_rec


@torch.no_grad()
def learn_conceptbank(model, concept_list, scenario, **kwargs):
    concept_dict = {}
    for concept in tqdm(concept_list):
        # Note: You can try other forms of prompting, e.g. "photo of {concept}" etc. here.
        text = clip.tokenize(f"{concept}").to(kwargs['device'])
        text_features = model.encode_text(text).cpu().numpy()
        text_features = text_features/np.linalg.norm(text_features)
        # store concept vectors in a dictionary. Adding the additional terms to be consistent with the
        # `ConceptBank` class (see `concepts/concept_utils.py`).
        concept_dict[concept] = (text_features, None, None, 0, {})

    print(f"# concepts: {len(concept_dict)}")
    concept_dict_path = os.path.join(kwargs['out_dir'], f"multimodal_concept_{kwargs['backbone_name']}_{scenario}_recurse:{kwargs['recurse']}.pkl")
    pickle.dump(concept_dict, open(concept_dict_path, 'wb'))
    print(f"Dumped to : {concept_dict_path}")


def get_concepts_multimodal(**kwargs):
    clip_backbone_name = kwargs['backbone_name'].split("-")[1]
    model, _ = clip.load(clip_backbone_name, device=kwargs['device'], download_root=kwargs['out_dir'])
    
    if kwargs['classes'] == "cifar10":
        # Pull CIFAR10 to get the class names.
        from torchvision import datasets
        cifar10_ds = datasets.CIFAR10(root=kwargs['out_dir'], train=True, download=True)
        # Get the class names.
        all_classes = list(cifar10_ds.classes)
        # Get the names of all concepts.
        all_concepts = get_concept_data(all_classes)
        # Clean the concepts for uniques, plurals etc. 
        all_concepts = clean_concepts(all_concepts, kwargs['out_dir'].split("/")[0])     
        all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # If we'd like to recurse in the conceptnet graph, specify `recurse > 1`.
        for i in range(1, kwargs['recurse']):
            all_concepts = get_concept_data(all_concepts)
            all_concepts = list(set(all_concepts))
            all_concepts = clean_concepts(all_concepts)
            all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # Generate the concept bank.
        learn_conceptbank(model, all_concepts, kwargs['classes'], **kwargs)
        
    elif kwargs['classes'] == "cifar100":
        from torchvision import datasets
        cifar100_ds = datasets.CIFAR100(root=kwargs['out_dir'], train=True, download=True)
        all_classes = list(cifar100_ds.classes)
        all_concepts = get_concept_data(all_classes)
        all_concepts = clean_concepts(all_concepts)
        all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # If we'd like to recurse in the conceptnet graph, specify `recurse > 1`.
        for i in range(1, kwargs['recurse']):
            all_concepts = get_concept_data(all_concepts)
            all_concepts = list(set(all_concepts))
            all_concepts = clean_concepts(all_concepts)
            all_concepts = list(set(all_concepts).difference(set(all_classes)))
        learn_conceptbank(model, all_concepts, kwargs['classes'], **kwargs)

    else:
        raise ValueError(f"Unknown classes: {kwargs['classes']}. Define your dataset here!")
