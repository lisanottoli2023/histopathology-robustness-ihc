import argparse
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataloader import get_dataloaders
from src.models.model import build_model
from src.evaluation.evaluator import  Evaluator
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model for histopathology image classification")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--model", type=str, required=True, help="Model architecture to choose", choices=["resnet50", "efficientnet_b3", "vit_base_patch16_224"])
    parser.add_argument("--experiment", type=str, default="all_stains_combined", help="Experiment name")
    parser.add_argument("--output_dir", type=str, help="Directory to save evaluation results")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(model_override=args.model, experiment_override=args.experiment)
    set_seed( cfg.training.seed)
    _, _, test_loader = get_dataloaders(cfg)
    model = build_model(cfg)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    evaluator = Evaluator(model, test_loader, cfg)
    results_df = evaluator.evaluate_all()
    stain_df = evaluator.evaluate_by_stain(results_df)
    stain_df.to_csv(f"{args.output_dir}/tables/per_stain_metrics.csv", index=False)

if __name__ == "__main__":
    main()