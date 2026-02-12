from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


def metrics_report(X, y, model, domain_name, class_names=None):
    y_pred = model.predict(X)
    bal_acc = balanced_accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    recall_macro = recall_score(y, y_pred, average="macro")

    cm = confusion_matrix(y, y_pred)
    specificity = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp))
    spec_macro = np.mean(specificity)
    
    return {
        "Domain": domain_name,
        "Labels": np.unique(y) if class_names is None else class_names, 
        "Balanced_Accuracy": bal_acc,
        "F1_Macro": f1_macro,
        "Recall_Macro": recall_macro,
        "Specificity_Macro": spec_macro,
        "Confusion Matrix": cm
    }

def print_metrics(metrics, logs=False, save=False, path=None):
    domain, class_names, b_acc, f1, recall, spec, cm = metrics["Domain"], metrics["Labels"], metrics["Balanced_Accuracy"], metrics["F1_Macro"], metrics["Recall_Macro"], metrics["Specificity_Macro"], metrics["Confusion Matrix"]

    # Print metrics information
    text = (
        f"Domain: {domain}\n"
        f"Balanced Accuracy: {b_acc:.4f}\n"
        f"F1 Macro: {f1:.4f}\n"
        f"Recall Macro: {recall:.4f}\n"
        f"Specificity Macro: {spec:.4f}\n"
    )

    if not logs:
        print("\n===== Metrics =====")
        print(text)
    else:
        assert path is not None, "Path must be provided when logs=True"
        path = Path(path)
        json_path = path.with_suffix(".json")

        with open(json_path, "w") as f:
            json.dump(
                {
                    "domain": domain,
                    "balanced_accuracy": b_acc,
                    "f1_macro": f1,
                    "recall_macro": recall,
                    "specificity_macro": spec,
                },
                f,
                indent=2
            )

    # Plot confusion metrics with labels
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
    