
import torch
from collections import Counter 
from sklearn.metrics import f1_score
from pathlib import Path 
from utils.mlflow_utils import log_metrics



class Trainer: 
    def __init__(self, model, train_loader, val_loader, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.min_lr)
        self.best_macro_f1 = 0.0
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        if cfg.training.class_weights:
            labels = [s[1] for s in train_loader.dataset.samples]
            counts = Counter(labels)
            weights = [1.0 / counts[i] for i in range(cfg.model.num_classes)]
            weights = torch.tensor(weights, dtype=torch.float).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()


    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels, _ in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        avg_loss = total_loss / total
        accuracy = correct / total
        return {"loss": avg_loss, "accuracy": accuracy}


    def _validate_epoch(self, epoch):
        total_loss = 0.0
        correct = 0
        total = 0
        macro_f1 = 0.0
        all_preds = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = macro_f1
                checkpoint_path = Path(self.cfg.checkpoints.directory) / f"{self.cfg.model.architecture}_best.pt"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Saved best model with macro F1: {macro_f1:.4f}")
            avg_loss = total_loss / total
            accuracy = correct / total
        return {"loss": avg_loss, "accuracy": accuracy, "macro_f1": macro_f1}

    def train(self):
        patience_counter = 0
        for epoch in range(self.cfg.training.epochs):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)
            print(f"Epoch {epoch+1}/{self.cfg.training.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val Macro F1: {val_metrics['macro_f1']:.4f}")
            self.scheduler.step()
            log_metrics(train_metrics, step=epoch, prefix="train")
            log_metrics(val_metrics, step=epoch, prefix="val")
            if val_metrics['macro_f1'] > self.best_macro_f1:
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.cfg.training.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        return self.best_macro_f1
    