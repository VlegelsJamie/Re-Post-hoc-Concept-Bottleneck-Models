from torchvision import datasets
import torch
import os


def get_dataset(preprocess=None, **kwargs):
    if kwargs['dataset'] == "cifar10":
        trainset = datasets.CIFAR10(root=kwargs['out_dir'], train=True,
                                    download=True, transform=preprocess)
        testset = datasets.CIFAR10(root=kwargs['out_dir'], train=False,
                                    download=True, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=kwargs['batch_size'],
                                              shuffle=True, num_workers=kwargs['num_workers'])
        test_loader = torch.utils.data.DataLoader(testset, batch_size=kwargs['batch_size'],
                                          shuffle=False, num_workers=kwargs['num_workers'])
    
    elif kwargs['dataset'] == "cifar100":
        trainset = datasets.CIFAR100(root=kwargs['out_dir'], train=True,
                                    download=True, transform=preprocess)
        testset = datasets.CIFAR100(root=kwargs['out_dir'], train=False,
                                    download=True, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=kwargs['batch_size'],
                                              shuffle=True, num_workers=kwargs['num_workers'])
        test_loader = torch.utils.data.DataLoader(testset, batch_size=kwargs['batch_size'],
                                          shuffle=False, num_workers=kwargs['num_workers'])

    elif kwargs['dataset'] == "cub":
        from .cub import load_cub_data
        from .constants import CUB_PROCESSED_DIR, CUB_DATA_DIR
        from torchvision import transforms
        num_classes = 200
        TRAIN_PKL = os.path.join(CUB_PROCESSED_DIR, "train.pkl")
        TEST_PKL = os.path.join(CUB_PROCESSED_DIR, "test.pkl")
        normalizer = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        train_loader = load_cub_data([TRAIN_PKL], use_attr=False, no_img=False, 
            batch_size=kwargs['batch_size'], uncertain_label=False, image_dir=CUB_DATA_DIR, resol=224, normalizer=normalizer,
            n_classes=num_classes, resampling=True)

        test_loader = load_cub_data([TEST_PKL], use_attr=False, no_img=False, 
                batch_size=kwargs['batch_size'], uncertain_label=False, image_dir=CUB_DATA_DIR, resol=224, normalizer=normalizer,
                n_classes=num_classes, resampling=True)

        classes = open(os.path.join(CUB_DATA_DIR, "classes.txt")).readlines()
        classes = [a.split(".")[1].strip() for a in classes]
        idx_to_class = {i: classes[i] for i in range(num_classes)}
        classes = [classes[i] for i in range(num_classes)]
        print(len(classes), "num classes for cub")
        print(len(train_loader.dataset), "training set size")
        print(len(test_loader.dataset), "test set size")

    elif kwargs['dataset'] == "ham10000":
        from .derma_data import load_ham_data
        train_loader, test_loader, idx_to_class = load_ham_data(preprocess, **kwargs)
        class_to_idx = {v:k for k,v in idx_to_class.items()}
        classes = list(class_to_idx.keys())

    elif kwargs['dataset'] == "coco":
        from .coco import load_coco_data
        train_loader, test_loader, idx_to_class = load_coco_data(preprocess, **kwargs)
        class_to_idx = {v:k for k,v in idx_to_class.items()}
        classes = list(class_to_idx.keys())

    elif kwargs['dataset'] == "isic":
        from .isic import load_isic_data
        train_loader, test_loader, idx_to_class = load_isic_data(preprocess, **kwargs)
        class_to_idx = {v:k for k,v in idx_to_class.items()}
        classes = list(class_to_idx.keys())

    elif "metashift" in kwargs['dataset']:
        from .metashift import load_metashift_data
        scenario = kwargs['dataset'].split("_")[1]
        train_loader, test_loader, idx_to_class, classes = load_metashift_data(preprocess, scenario, **kwargs)
        class_to_idx = {v:k for k,v in idx_to_class.items()}
        classes = list(class_to_idx.keys())

    else:
        raise ValueError(kwargs['dataset'])

    return train_loader, test_loader, idx_to_class, classes

