import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


def metrics_report(X, y, model, domain_name):
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
        "Labels": np.unique(y), 
        "Balanced_Accuracy": bal_acc,
        "F1_Macro": f1_macro,
        "Recall_Macro": recall_macro,
        "Specificity_Macro": spec_macro,
        "Confusion Matrix": cm
    }

def print_metrics(metrics):
    domain, class_names, b_acc, f1, recall, spec, cm = metrics["Domain"], metrics["Labels"], metrics["Balanced_Accuracy"], metrics["F1_Macro"], metrics["Recall_Macro"], metrics["Specificity_Macro"], metrics["Confusion Matrix"]

    # Print metrics information
    print(f"Printing metrics for Domain: {domain}")
    print(f"Balanced Accuracy: {b_acc}")
    print(f"F1 Macro: {f1}")
    print(f"Recall Macro: {recall}")
    print(f"Specificity Macro: {spec}")

    # Plot confusion metrics with labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)

    plt.title(f"Confusion Matrix — {domain}", fontsize=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()
    