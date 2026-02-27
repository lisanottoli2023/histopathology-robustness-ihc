Plan: Histopathology IHC Robustness Analysis — Project Architecture & Implementation
Context
This project evaluates robustness of a multi-class histopathology classification model across
12 IHC stains (CD20, CD3, CD34, CD38, CD68, CDK4, D2, FAP, P53, SMA, cyclin, ki67).
The core experiment is: train a model on all stains combined → evaluate performance broken
down per stain to measure stain-specific degradation.

Deliverables: reproducible codebase + interactive Jupyter notebooks.
MLOps: MLflow (local). Backbones: ResNet-50, EfficientNet-B3, ViT-B/16 (benchmarked).

Dataset: MIHIC at /home/lisa/Downloads/MIHIC_dataset/dataset/ (train/test/val splits).

309,698 images total; 7 classes (Tumor, Stroma, Immune cells, Necrosis, Other, alveoli, background)
Stain is embedded in every filename: {patient_id}-{stain}-{section}-{tile}.png
Strong class imbalance: Tumor (~91K train) >> Immune cells (~6.6K train)
Proposed Directory Structure

histopathology-robustness-ihc/
├── config/
│   ├── base.yaml                      # data, training, MLflow, report defaults
│   ├── models/
│   │   ├── resnet50.yaml
│   │   ├── efficientnet.yaml
│   │   └── vit.yaml
│   └── experiments/
│       └── all_stains_combined.yaml
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # MIHICDataset — stain parsed from filename
│   │   ├── transforms.py              # train/val augmentation pipelines
│   │   └── dataloader.py             # DataLoader factory
│   ├── model/
│   │   ├── __init__.py
│   │   └── classification.py          # model factory (ResNet/EfficientNet/ViT via timm)
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py                 # training loop + MLflow logging
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py               # per-stain evaluation (single inference pass)
│   │   ├── metrics.py                 # accuracy, macro F1, per-class metrics
│   │   └── visualizer.py              # confusion matrices, bar charts, GradCAM
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seed.py                    # PyTorch + NumPy + Python seed fixing
│   │   ├── mlflow_utils.py            # MLflow run setup helpers
│   │   └── config.py                  # OmegaConf load + merge (no Hydra)
│   └── notebook/
│       ├── eda.ipynb                  # EXISTING — extend with stain distribution
│       └── robustness_report.ipynb    # NEW — primary deliverable
│
├── scripts/
│   ├── train.py                       # CLI: --model --experiment --seed --run-name
│   └── evaluate.py                    # CLI: --checkpoint --model --output-dir
│
├── mlops/
│   └── mlruns/                        # MLflow tracking store (auto-created)
│
├── checkpoints/
│   └── {arch}_{timestamp}/
│       ├── best_model.pt
│       └── config_snapshot.yaml
│
└── reports/
    ├── figures/
    │   ├── confusion_matrices/
    │   ├── per_stain_bars/
    │   └── gradcam/
    └── tables/
        └── per_stain_metrics_{arch}_{date}.csv

The question is: does the model correctly identify "Tumor" or "Immune cells" regardless of which stain was used?

two levels of finding : 
which model is best overall ? (metrics)
which model is most robust? 


ResNet-50 — the baseline

The "standard" backbone in medical imaging research
Well studied on histopathology, lots of published results to compare against
Gives you a solid reference point
EfficientNet-B3 — the efficient CNN

Better accuracy/parameter ratio than ResNet
Still a CNN, so you can isolate whether architecture efficiency changes robustness
Common in recent histopathology papers
ViT-B/16 — the architectural wildcard

Completely different paradigm: attention-based, no convolutions
CNNs are biased toward local textures — which is exactly what IHC stains affect
Transformers capture more global patterns, so the hypothesis is: maybe they're more stain-robust because they rely less on local color/texture cues
The 3 models together let you answer:

Is robustness a property of the architecture family (CNN vs Transformer), or of the specific model?


