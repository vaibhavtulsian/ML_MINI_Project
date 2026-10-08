import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from typing import Dict, Any

class Evaluator:
    def __init__(self, config_path: str = "configs/config.yaml"):
        import yaml
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.figures_dir = self.config["outputs"]["figures_dir"]
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def evaluate_model(self, model_name: str, target_name: str, y_true, y_pred, y_prob) -> Dict[str, Any]:
        """
        Generates and prints evaluation metrics for a single model and target.
        """
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        
        print(f"\n{'='*40}")
        print(f"Target: {target_name} | Model: {model_name}")
        print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        
        self._plot_confusion_matrix(model_name, target_name, y_true, y_pred)
        self._plot_roc_pr_curves(model_name, target_name, y_true, y_prob)
        
        return {
            "auroc": auroc,
            "auprc": auprc,
            "report": classification_report(y_true, y_pred, output_dict=True)
        }
        
    def _plot_confusion_matrix(self, model_name, target_name, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{model_name} Confusion Matrix - {target_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f"{model_name}_{target_name}_cm.png"), dpi=300)
        plt.close()
        
    def _plot_roc_pr_curves(self, model_name, target_name, y_true, y_prob):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax1.plot(fpr, tpr, label=f'{model_name} (AUROC = {roc_auc_score(y_true, y_prob):.3f})')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve - {target_name}')
        ax1.legend()
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax2.plot(recall, precision, label=f'{model_name} (AUPRC = {average_precision_score(y_true, y_prob):.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve - {target_name}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f"{model_name}_{target_name}_curves.png"), dpi=300)
        plt.close()
