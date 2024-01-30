import torch
import torch.nn as nn
import numpy as np

from torchvision import models, transforms


class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def get_model(backbone_name="resnet18_cub", full_model=False, **kwargs):
    if "clip" in backbone_name:
        import clip
        # We assume clip models are passed of the form - clip-RN50
        clip_backbone_name = backbone_name.split("-")[1]
        backbone, preprocess = clip.load(clip_backbone_name, device=kwargs['device'], download_root=kwargs['out_dir'])
        backbone = backbone.eval()
        model = None
    
    elif backbone_name == "resnet18_cub":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        model = ptcv_get_model(backbone_name, pretrained=True, root=kwargs['out_dir'])
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        cub_mean_pxs = np.array([0.5, 0.5, 0.5])
        cub_std_pxs = np.array([2., 2., 2.])
        preprocess = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cub_mean_pxs, cub_std_pxs)
            ])
    
    elif backbone_name.lower() == "ham10000_inception":
        from .derma_models import get_derma_model
        model, backbone, model_top = get_derma_model(backbone_name.lower(), **kwargs)
        preprocess = transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      ])
    
    elif backbone_name == "resnet50":
        # Load the pretrained ResNet50 model
        model = models.resnet50(pretrained=True)

        # Copy the model to create the backbone
        backbone = models.resnet50(pretrained=True)
        # Remove the last fully connected layer to get the backbone
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))

        # Preprocessing steps
        preprocess = transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    else:
        raise ValueError(backbone_name)

    if full_model:
        return model, backbone, preprocess
    else:
        return backbone, preprocess



