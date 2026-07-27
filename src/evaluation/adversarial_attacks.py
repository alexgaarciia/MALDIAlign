import pandas as pd

from sklearn.metrics import balanced_accuracy_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def clamp01(x):
    """Projects a tensor to the valid range [0, 1] element-wise.

    In MALDI-TOF spectra normalized with row_minmax_normalize, all values
    lie in [0, 1]. After adding an adversarial perturbation, some bins may
    exceed this range — this function clips them back.

    Args:
        x (torch.Tensor): tensor of any shape.

    Returns:
        torch.Tensor: tensor with all values clipped to [0, 1].
    """
    return torch.clamp(x, 0.0, 1.0)

def fgsm_attack(model, x, y, eps):
    """Computes a single-step Fast Gradient Sign Method (FGSM) adversarial example.

    FGSM perturbs the input in the direction that maximally increases the
    cross-entropy loss with respect to the true label, within an L-inf ball
    of radius eps. Being a single-step method, it is fast but weaker than
    iterative attacks.

    Args:
        model (nn.Module): the classifier to attack (must be in eval mode).
        x (torch.Tensor): clean input batch, shape (B, input_dim), in [0, 1].
        y (torch.Tensor): true labels, shape (B,).
        eps (float): L-inf perturbation budget.

    Returns:
        torch.Tensor: adversarial examples x_adv, same shape as x, in [0, 1].
    """
    x_adv = x.clone().detach().requires_grad_(True)
    loss  = F.cross_entropy(model(x_adv), y)
    grad  = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv + eps * grad.sign()
    return clamp01(x_adv.detach())

def pgd_attack(model, x, y, eps, alpha, steps, random_start=True):
    """Computes a multi-step Projected Gradient Descent (PGD) adversarial example.

    PGD iteratively applies FGSM steps of size alpha, projecting back onto
    the L-inf ball of radius eps around the original input after each step.
    It is considered the gold-standard first-order attack and is significantly
    stronger than FGSM for the same eps budget.

    Args:
        model (nn.Module): the classifier to attack (must be in eval mode).
        x (torch.Tensor): clean input batch, shape (B, input_dim), in [0, 1].
        y (torch.Tensor): true labels, shape (B,).
        eps (float): L-inf perturbation budget (radius of the allowed ball).
        alpha (float): step size per iteration. Typically eps/4.
        steps (int): number of PGD iterations.
        random_start (bool): if True, initializes x_adv from a random point
            inside the L-inf ball instead of x itself. Helps escape local
            optima and produces stronger attacks. Default: True.

    Returns:
        torch.Tensor: adversarial examples x_adv, same shape as x, in [0, 1].
    """
    x_orig = x.detach()
    if random_start:
        x_adv = clamp01(x_orig + (2 * torch.rand_like(x_orig) - 1.0) * eps)
    else:
        x_adv = x_orig.clone()

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        loss  = F.cross_entropy(model(x_adv), y)
        grad  = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha * grad.sign()
        # Project back onto the L-inf ball around x_orig
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = clamp01(x_orig + delta)

    return x_adv.detach()

def eval_under_attack(model, X_te, y_te, attack_fn, device, batch_size=256):
    """Evaluates a model's balanced accuracy under a given adversarial attack.

    Iterates over the test set in batches, generates adversarial examples
    for each batch using attack_fn, and measures the balanced accuracy of
    the model on those adversarial examples.

    Args:
        model (nn.Module): trained classifier.
        X_te (np.ndarray): test features, shape (N, input_dim), in [0, 1].
        y_te (np.ndarray): test labels, shape (N,), integer-encoded.
        attack_fn (callable): function (x_batch, y_batch) -> x_adv.
            Pass lambda x, y: x for clean evaluation (no attack).
        device (torch.device): device for computation.
        batch_size (int): batch size for the DataLoader. Default: 256.

    Returns:
        float: balanced accuracy score on adversarial examples.
    """
    model.eval()
    all_preds, all_true = [], []

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_te, dtype=torch.float32),
            torch.tensor(y_te, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=False
    )

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_adv = attack_fn(x_batch, y_batch)
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(y_batch.cpu().numpy())

    return balanced_accuracy_score(all_true, all_preds)

def sweep_attacks(models_dict, X_te, y_te, eps_list,
                  pgd_steps_list=(10, 40), device="cpu"):
    """Sweeps multiple adversarial attacks across a range of epsilon values
    for each model, returning a DataFrame with balanced accuracy results.

    For each model and each epsilon, evaluates:
        - Clean accuracy (no attack, only at eps=0)
        - FGSM
        - PGD with each number of steps in pgd_steps_list

    The PGD step size alpha is set to eps/4 following standard practice.

    Args:
        models_dict (dict): mapping of model name (str) -> nn.Module.
        X_te (np.ndarray): test features, shape (N, input_dim), in [0, 1].
        y_te (np.ndarray): test labels, shape (N,), integer-encoded.
        eps_list (list of float): L-inf perturbation budgets to evaluate.
            Should include 0.0 for clean accuracy.
        pgd_steps_list (tuple of int): PGD iteration counts to evaluate.
            Default: (10, 40).
        device (torch.device or str): device for computation. Default: "cpu".

    Returns:
        pd.DataFrame: columns [model, attack, epsilon, balanced_accuracy].
    """
    rows = []

    for model_name, model in models_dict.items():
        print(f"\n--- Model: {model_name} ---")

        # Clean (no attack)
        ba_clean = eval_under_attack(
            model, X_te, y_te,
            attack_fn=lambda x, y: x,
            device=device,
        )
        rows.append({"model": model_name, "attack": "Clean",
                     "epsilon": 0.0, "balanced_accuracy": ba_clean})
        print(f"  Clean BA: {ba_clean:.4f}")

        for eps in eps_list:
            if eps == 0.0:
                continue

            # FGSM
            ba_fgsm = eval_under_attack(
                model, X_te, y_te,
                attack_fn=lambda x, y, e=eps: fgsm_attack(model, x, y, e),
                device=device,
            )
            rows.append({"model": model_name, "attack": "FGSM",
                         "epsilon": eps, "balanced_accuracy": ba_fgsm})
            print(f"  ε={eps:.3f} FGSM  BA={ba_fgsm:.4f}")

            # PGD with varying number of steps
            for steps in pgd_steps_list:
                alpha = eps / 4
                ba_pgd = eval_under_attack(
                    model, X_te, y_te,
                    attack_fn=lambda x, y, e=eps, a=alpha, s=steps: pgd_attack(
                        model, x, y, e, a, s, random_start=True),
                    device=device,
                )
                rows.append({"model": model_name, "attack": f"PGD-{steps}",
                             "epsilon": eps, "balanced_accuracy": ba_pgd})
                print(f"  ε={eps:.3f} PGD-{steps} BA={ba_pgd:.4f}")

    return pd.DataFrame(rows)
