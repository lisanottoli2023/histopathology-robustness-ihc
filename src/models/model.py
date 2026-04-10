import timm
import torch 

SUPPORTED_ARCHITECTURES = ["resnet50", "efficientnet_b3", "vit_base_patch16_224"]


def build_model(cfg):
    if cfg.model.architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Model architecture '{cfg.model.architecture}' is not supported. Please choose from: {SUPPORTED_ARCHITECTURES}")
    model = timm.create_model(
        cfg.model.architecture, 
        pretrained=cfg.model.pretrained, 
        num_classes=cfg.model.num_classes, 
        drop_rate=cfg.model.dropout
    )
    print(f"Model '{cfg.model.architecture}' created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.")
    return model


