import argparse
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataloader import get_dataloaders
from src.models.model import build_model
from src.utils.mlflow_utils import log_config, setup_mlflow
from src.training.trainer import Trainer
import mlflow

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for histopathology image classification")
    parser.add_argument("--model", type=str, required=True, help="Model architecture to choose", choices=["resnet50", "efficientnet_b3", "vit_base_patch16_224"])
    parser.add_argument("--experiment", type=str, default="all_stains_combined", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(model_override=args.model, experiment_override=args.experiment)
    set_seed(args.seed)
    train_loader ,val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg)
    run = setup_mlflow(cfg)
    log_config(cfg)
    Trainer(model, train_loader, val_loader, cfg).train()
    mlflow.end_run()

if __name__ == "__main__":
    main()

