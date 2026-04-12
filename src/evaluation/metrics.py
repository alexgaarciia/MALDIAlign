import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder


def metrics_report(X, y, model, domain_name, class_names=None, compute_pr=False):
    """
    Evaluates classifier and returns a dictionary of performance metrics.

    Calculates balanced accuracy, macro F1, macro recall, macro specificity, and ROC-AUC. 
    Specificity is calculated manually per class and averaged. ROC-AUC is computed 
    using a 'One-vs-Rest' strategy for multi-class scenarios.

    Args:
        X (array-like): Feature matrix for evaluation.
        y (array-like): Ground truth target values.
        model (object): A trained classifier with `predict` and `predict_proba` methods.
        domain_name (str): Name or identifier of the dataset domain.
        class_names (list, optional): Human-readable names for the classes.
        compute_pr (bool): Whether to calculate the Precision-Recall AUC (PR-AUC).

    Returns:
        dict: A dictionary containing the domain name, labels, various macro-averaged 
              scores, and the raw confusion matrix.
    """

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    bal_acc = balanced_accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y, y_pred)
    specificity = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp))
    spec_macro = np.mean(specificity)

    # ROC-AUC
    n_classes = y_proba.shape[1]
    try:
        classes_in_y = np.unique(y)
        if len(classes_in_y) < 2:
            roc_auc = np.nan
        elif n_classes == 2:
            roc_auc = roc_auc_score(y, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='macro', labels=class_names)
    except ValueError:
        roc_auc = np.nan

    # PR-AUC
    if compute_pr:
        try:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            y_true_onehot = np.zeros_like(y_proba)
            y_true_onehot[np.arange(len(y_encoded)), y_encoded] = 1
            
            pr_aucs = []
            for c in range(n_classes):
                if y_true_onehot[:, c].sum() > 0:
                    pr_aucs.append(average_precision_score(y_true_onehot[:, c], y_proba[:, c]))
            pr_auc = np.mean(pr_aucs) if pr_aucs else np.nan

        except ValueError:
            pr_auc = np.nan
    else:
        pr_auc = None
    
    return {
        "Domain": domain_name,
        "Labels": np.unique(y) if class_names is None else class_names,
        "Balanced_Accuracy": bal_acc,
        "F1_Macro": f1_macro,
        "Recall_Macro": recall_macro,
        "Specificity_Macro": spec_macro,
        "ROC_AUC_Macro": roc_auc,
        "PR_AUC_Macro": pr_auc,
        "Confusion Matrix": cm
    }


def metrics_report_mlp(dataloader, model, domain_name, device="cpu", class_names=None, compute_pr=True):
    """
    Evaluates a PyTorch MLP model using a DataLoader and returns performance metrics.

    Sets the model to evaluation mode and performs a forward pass over the dataset 
    to aggregate predictions and probabilities. 

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader providing the evaluation data.
        model (torch.nn.Module): The PyTorch neural network to evaluate.
        domain_name (str): Name or identifier of the dataset domain.
        device (str): The device (e.g., "cpu", "cuda") on which to perform computation.
        class_names (list, optional): Human-readable names for the classes.
        compute_pr (bool): Whether to calculate the Precision-Recall AUC (PR-AUC).

    Returns:
        dict: A dictionary containing the domain name, labels, macro-averaged 
              scores, and the raw confusion matrix.
    """

    model.eval()
    model.to(device)

    y_true = []
    y_pred = []
    y_proba = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1) 

            y_true.append(y.numpy())
            y_pred.append(preds.cpu().numpy())
            y_proba.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_proba = np.vstack(y_proba)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    specificity = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp))
    spec_macro = np.mean(specificity)

    # ROC-AUC
    n_classes = y_proba.shape[1]
    try:
        if len(np.unique(y_true)) < 2:
            roc_auc = np.nan
        elif n_classes == 2:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    except ValueError:
        roc_auc = np.nan
    
    # PR-AUC
    if compute_pr:
        try:
            y_true_onehot = np.zeros_like(y_proba)
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            
            pr_aucs = []
            for c in range(n_classes):
                if y_true_onehot[:, c].sum() > 0:
                    pr_aucs.append(average_precision_score(y_true_onehot[:, c], y_proba[:, c]))
            pr_auc = np.mean(pr_aucs) if pr_aucs else np.nan

        except ValueError:
            pr_auc = np.nan
    else:
        pr_auc = None

    
    return {
        "Domain": domain_name,
        "Labels": np.unique(y_true) if class_names is None else class_names,
        "Balanced_Accuracy": bal_acc,
        "F1_Macro": f1_macro,
        "Recall_Macro": recall_macro,
        "Specificity_Macro": spec_macro,
        "ROC_AUC_Macro": roc_auc,
        "PR_AUC_Macro": pr_auc,
        "Confusion Matrix": cm
    }



def print_metrics(metrics, logs=False, save=False, path=None, show_cm=True, ignore_pr=True):
    """
    Outputs the metrics report and handles file logging and visualization.

    This function can print summary statistics, save metrics to a JSON file, and 
    render or save a Confusion Matrix plot using Matplotlib.

    Args:
        metrics (dict): The metrics dictionary produced by metrics_report or metrics_report_mlp.
        logs (bool): If True, saves the numeric metrics to a JSON file. Requires `path`.
        save (bool): If True, saves the Confusion Matrix plot to a PNG file. Requires `path`.
        path (str or Path, optional): Destination file path for saving logs or plots.
        show_cm (bool): Whether to display the Confusion Matrix plot window.
        ignore_pr (bool): If True, PR-AUC will be omitted from the printed/logged output.

    Raises:
        AssertionError: If `logs` or `save` is True but `path` is not provided.
    """

    domain, class_names, b_acc, f1, recall, spec, roc, pr, cm = metrics["Domain"], metrics["Labels"], metrics["Balanced_Accuracy"], metrics["F1_Macro"], metrics["Recall_Macro"], metrics["Specificity_Macro"], metrics["ROC_AUC_Macro"], metrics["PR_AUC_Macro"], metrics["Confusion Matrix"]

    text = (
        f"Domain: {domain}\n"
        f"Balanced Accuracy: {b_acc:.4f}\n"
        f"F1 Macro: {f1:.4f}\n"
        f"Recall Macro: {recall:.4f}\n"
        f"Specificity Macro: {spec:.4f}\n"
        f"ROC-AUC Macro: {roc:.4f}\n"
    ) 

    if not ignore_pr:
        text += f"PR-AUC Macro: {pr:.4f}\n"

    if not logs:
        print("\n===== Metrics =====")
        print(text)
    else:
        assert path is not None, "Path must be provided when logs=True"
        path = Path(path)
        json_path = path.with_suffix(".json")

        out_dict = {
            "domain": domain,
            "balanced_accuracy": b_acc,
            "f1_macro": f1,
            "recall_macro": recall,
            "specificity_macro": spec,
            "roc_auc_macro": roc,
        }
        if not ignore_pr:
            out_dict["pr_auc_macro"] = pr

        with open(json_path, "w") as f:
            json.dump(out_dict, f, indent=2)

    if show_cm:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")

        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)

        plt.title(f"Confusion Matrix — {domain}", fontsize=12)
        plt.xlabel("") 
        plt.ylabel("")  
        plt.tight_layout()

        if save:
            assert path is not None, "Path must be provided when save=True"
            fig_path = Path(path).with_suffix(".png")
            plt.savefig(fig_path)
            plt.close()
        else:
            plt.show()


def compute_per_antibiotic_auc(logits, labels, antibiotic_names=None):
    """
    Computes individual ROC-AUC scores for every label column in a multi-label task.

    This provides a view of performance by calculating a separate AUC 
    for each specific antibiotic or category.

    Args:
        logits (torch.Tensor): Raw model outputs (pre-sigmoid).
        labels (torch.Tensor): Ground truth binary labels, potentially containing NaNs.
        antibiotic_names (list, optional): List of names corresponding to the 
                                           label columns. Defaults to "ab_j" indexing.

    Returns:
        dict: A dictionary mapping each antibiotic name to its respective ROC-AUC score.
    """

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    auc_dict = {}

    for j in range(y_true.shape[1]):
        name = antibiotic_names[j] if antibiotic_names is not None else f"ab_{j}"

        mask = ~np.isnan(y_true[:, j])
        y_j = y_true[mask, j]
        p_j = probs[mask, j]

        if len(y_j) == 0 or len(np.unique(y_j)) < 2:
            auc_dict[name] = np.nan
            continue

        auc_dict[name] = float(roc_auc_score(y_j, p_j))

    return auc_dict
    

def compute_multilabel_auc(logits, labels):
    """
    Computes the macro-averaged ROC-AUC across multiple binary labels (e.g., antibiotics).

    The function handles multi-label data where some labels might be missing (NaN). 
    It applies a sigmoid activation to the logits and ignores label columns that 
    do not contain both positive and negative samples in the current batch.

    Args:
        logits (torch.Tensor): Raw model outputs (pre-sigmoid).
        labels (torch.Tensor): Ground truth binary labels, potentially containing NaNs.

    Returns:
        float: The mean ROC-AUC score across all valid labels, or np.nan if no 
               valid labels were found.
    """
    
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    aucs = []

    for j in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, j])
        y_j = y_true[mask, j]
        p_j = probs[mask, j]

        if len(np.unique(y_j)) < 2:
            continue

        aucs.append(roc_auc_score(y_j, p_j))

    if len(aucs) == 0:
        return np.nan

    return float(np.mean(aucs))
