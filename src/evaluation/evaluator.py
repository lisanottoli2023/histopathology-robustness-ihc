import torch
import pandas as pd
from .metrics import compute_metrics



class Evaluator: 
    def __init__(self, model,test_loader, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = cfg.data.class_list
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.cfg = cfg

    def evaluate_all(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_stains = []
        with torch.no_grad():
            for images, labels, stains in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_stains.extend(stains)
        results_df = pd.DataFrame({
            "true_label": all_labels,
            "pred_label": all_preds,
            "stain": all_stains, 
            "correct": [pred == true for pred, true in zip(all_preds, all_labels)]
        })
        return results_df
    
    def evaluate_by_stain(self, results_df):
        stain_groups = results_df.groupby("stain")
        rows = []
        for stain, group in stain_groups:
            metrics = compute_metrics(group["true_label"], group["pred_label"], self.class_names)
            rows.append({
                "stain": stain,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            })
        for class_name, class_metrics in metrics["per_class"].items():
            rows[f"{class_name}_f1"] = class_metrics["f1"]
        rows.append(rows)
        return pd.DataFrame(rows)



