# MALDIAlign

**Biologically Informed Representation Learning for Robust Cross-Center Generalization of MALDI-TOF Mass Spectrometry**

This repository contains the official implementation of **DALMA** (*Domain ALignment for MALDI-TOF MS*), a probabilistic representation learning framework that learns biologically structured latent representations of MALDI-TOF mass spectra that generalize across heterogeneous clinical centers — enabling **zero-shot deployment on previously unseen sites**.

> García-Navarro, Sevilla-Salcedo, Rodríguez-Sánchez, Gómez-Verdejo. *Biologically Informed Representation Learning for Robust Cross-Center Generalization of MALDI-TOF Mass Spectrometry* (under review).

---

## Table of Contents

1. [Objective](#1-objective)
2. [What DALMA Does](#2-what-dalma-does)
3. [Models](#3-models)
4. [Repository Structure](#4-repository-structure)
5. [Installation & Setup](#5-installation--setup)
6. [Input Data Format](#6-input-data-format)
7. [Configuration](#7-configuration)
8. [How to Use It](#8-how-to-use-it)
9. [Downstream Tasks & Experiments](#9-downstream-tasks--experiments)
10. [Benchmark Datasets](#10-benchmark-datasets)
11. [Citation](#11-citation)

---

## 1. Objective

Machine learning models for MALDI-TOF mass spectrometry are highly effective for clinical microbiology tasks (microbial identification and antimicrobial resistance prediction), but their deployment across institutions is limited by **domain shift**: differences in instrumentation, calibration, sample preparation, preprocessing pipelines, and patient populations cause models to capture *technical artifacts* instead of *transferable biological information*.

The goal of this project is to learn a latent representation **z** of a spectrum that:

- **Preserves biological information** (species identity, resistance phenotype), and
- **Is robust to acquisition-specific variability** (the center/instrument/protocol where the spectrum was measured),

so that a model trained on a set of source centers can be deployed on a **new, unseen center without any adaptation** (zero-shot).

---

## 2. What DALMA Does

DALMA is a **probabilistic (VAE-based) representation learning framework** that explicitly disentangles biological signal from acquisition-specific variability. It combines four ideas:

| Component | Role |
|-----------|------|
| **Shared probabilistic encoder** `q(z\|x)` | Maps spectra from *all* centers into a common latent space. |
| **Domain-specific decoders** `{p(x\|z)}` | One decoder per training center absorbs acquisition-specific variability during reconstruction, keeping the shared latent space clean. |
| **Species-conditioned latent prior** `p(z\|s)` | Each species has its own learnable Gaussian; the KL term pulls spectra of the same species toward a shared distribution across centers. |
| **Auxiliary AMR head** (optional) | When a single label prior cannot be defined (multi-label AMR), biological information is injected through an auxiliary prediction head trained jointly. |

**Training-time only:** domain information (which center a spectrum comes from) is used *only during training*.

**Inference-time:** DALMA collapses to a **single shared encoder**. Every spectrum, from any center (seen or unseen), is mapped through the same encoder using the posterior mean `z* = μ(x)`. Domain-specific decoders are discarded.

This design lets the representation drop into existing MALDI-TOF workflows without any institution-specific component.

---

## 3. Models

All model classes live under [models/deep/](models/deep/) and [models/baselines/](models/baselines/) and are instantiated through the factory in [models/build_model.py](models/build_model.py) based on the `model.type` field in the config.

### DALMA and its ablation variants

| `model.type` | Class | Prior | Decoders | Notes |
|--------------|-------|-------|----------|-------|
| `vae_multidecoder_prior` | `MultiVAE_Bernoulli_SpeciesPrior_Extended` | Species-conditioned | Multi (per-domain) | **Full DALMA** for microbial identification. |
| `vae_multidecoder` | `MultiVAE_Bernoulli_Extended` | Standard `N(0,I)` | Multi | Ablation: no biological prior. |
| `vae_bernoulli` | `VAE_Bernoulli_Extended` | Standard or species (`use_species_prior`) | Single | Ablation: shared decoder. |
| `vae_multidecoder_prior_amr_head_z` | `MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ` | Species-conditioned + auxiliary AMR head | Multi | **DALMA for AMR** (auxiliary supervision on `z`). |
| `vae_multidecoder_prior_amr_head_zemb` | `MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZEmb` | + species embedding | Multi | AMR head conditioned on species embedding. |

### Baselines (for comparison)

| `model.type` | Class | Strategy |
|--------------|-------|----------|
| `vae_bernoulli` | `VAE_Bernoulli_Extended` | Standard VAE (unsupervised representation learning). |
| `dann` | `DANNFull_Extended` | Domain-Adversarial Neural Network (adversarial domain adaptation). |
| `vae_multidecoder_coral` | `MultiVAE_CORAL` | Multi-domain CORAL (distribution alignment). |
| `cvae` | `ConditionalVAE_Bernoulli_Extended` | Conditional VAE. |
| — | `MLPClassifier_Extended` ([models/baselines/mlp.py](models/baselines/mlp.py)) | Fully supervised MLP on raw spectra. |
| — | `LinearProbe_Extended` ([models/baselines/mlp_latent.py](models/baselines/mlp_latent.py)) | Linear probe trained on frozen latent representations. |

> The **Maldi Transformer** baseline (large-scale pretraining) is evaluated through [experiments/species_id/run_classification_mlp_maldi_transformer.py](experiments/species_id/run_classification_mlp_maldi_transformer.py).

### Default architecture

- Encoder: fully connected `[2048, 1024, 512]` → two linear heads (μ, log σ²).
- Latent dimensionality: **L = 64**.
- Each domain-specific decoder mirrors the encoder (`[512, 1024, 2048]`) with a sigmoid output of dimension `M`.
- ReLU activations; Bernoulli reconstruction likelihood (binary cross-entropy).
- Optimizer: Adam, lr `1e-4`, weight decay `1e-5`, batch size 128, early stopping on validation ELBO (patience 20). Log-variances clamped to `[-6, 6]`.

---

## 4. Repository Structure

```
MALDIAlign/
├── configs/                      # YAML experiment configurations
│   ├── config.yaml               # Global data-path registry (edit this first)
│   ├── vae_multidecoder_prior/   # DALMA (species identification)
│   ├── vae_multidecoder_prior_multiamr/  # DALMA (AMR)
│   ├── vae_multidecoder/         # ablations / CORAL baseline
│   ├── vae/ , cvae/ , dann/      # baselines
│   └── adversarial_training/     # adversarial variant
│
├── models/
│   ├── build_model.py            # model factory (maps config → class)
│   ├── deep/                     # DALMA and deep baselines (VAEs, DANN, CORAL)
│   └── baselines/                # MLP classifier, linear probe, AMR MLPs
│
├── src/
│   ├── config/loader.py          # YAML loader
│   ├── data/                     # io, dataset loaders, preprocessing, splits
│   ├── dataloaders/builders.py   # torch DataLoader construction
│   ├── training/                 # data_pipeline.py + training loop
│   ├── evaluation/               # metrics, latent encoding, AMR eval, similarity
│   ├── finetuning/               # few-shot species adaptation
│   └── visualization/            # plots (loss curves, t-SNE, confusion, AMR)
│
├── experiments/                  # runnable scripts + saved results
│   ├── run_experiment.py         # MAIN entrypoint: train a representation model
│   ├── species_id/               # microbial identification (linear probe)
│   ├── amr/                       # AMR classification
│   ├── finetuning/               # few-shot adaptation grids
│   ├── novelty_detection/        # OOD / selective prediction (GMM)
│   ├── spectra_similarity/       # cross-domain cosine similarity analysis
│   └── adversarial_attacks/      # robustness experiments
│
├── requirements.txt
└── README.md
```

---

## 5. Installation & Setup

**Requirements:** Python ≥ 3.10, PyTorch 2.x, and a CUDA-capable GPU (recommended; CPU also works).

```bash
git clone https://github.com/alexgaarciia/MALDIAlign.git
cd MALDIAlign

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The pinned [requirements.txt](requirements.txt) targets CUDA 12.8 wheels for `torch==2.10.0`. If you are on CPU or a different CUDA version, install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/) and then install the remaining scientific dependencies (`numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `PyYAML`, `lightgbm`).

**Important:** After installing, edit [configs/config.yaml](configs/config.yaml) so the data paths point to your local `.pkl` files (see [Input Data Format](#6-input-data-format)).

---

## 6. Input Data Format

DALMA operates on **preprocessed, binned spectra** — not raw MALDI-TOF acquisitions. Each dataset is stored as a Python **pickle** (`.pkl`) file containing a dictionary:

```python
{
    "data":  np.ndarray,   # shape (N, M) — N spectra × M spectral bins, float
    "label": np.ndarray,   # shape (N,)   — species name per spectrum (str)
    "meta":  list[dict] | pd.DataFrame,  # per-spectrum metadata (must include "hospital")
    "amr":   np.ndarray,   # (optional) shape (N,) or (N, A) — resistance labels {0,1,NaN}
    "antibiotics": list[str],  # (optional) column names for the amr matrix
}
```

### Key conventions

- **`data`** — spectra after the standardized preprocessing pipeline of Weis et al.: variance stabilization, smoothing, baseline correction, intensity thresholding, trimming to **2,000–20,000 m/z**, **3-Da binning**, and logarithmic scaling → **M = 6,000-dimensional** feature vectors. Values are min-max normalized per row at load time (`row_minmax`), placing them in `[0, 1]` so decoder outputs can be interpreted as Bernoulli parameters.
- **`meta`** — a records list / DataFrame. The **`hospital`** column identifies the acquisition **domain** (e.g. `"DRIAMS_A"`, `"MARISMA"`, `"RKI"`, `"MS-UMG"`) and is used to (a) select which centers to load and (b) route each spectrum to its domain-specific decoder during training. Optional columns like `year` and `agar` are used for filtering.
- **`label`** — species names such as `"Klebsiella_Pneumoniae"`, `"Escherichia_Coli"`, `"Staphylococcus_Aureus"`, `"Pseudomonas_Aeruginosa"`, `"Enterococcus_Faecium"`, `"Enterobacter_cloacae_complex"`.
- **`amr`** — a multi-label matrix of binary resistance phenotypes (`1` = resistant, `0` = susceptible, `NaN` = not tested). Only needed for AMR experiments.

Dataset-specific loaders live in [src/data/datasets.py](src/data/datasets.py) (`load_driams`, `load_marisma`, `load_msumg`, `load_rki`) and normalize the `hospital` column so all centers share the same schema.

> To run DALMA on **your own center**, produce a `.pkl` in the format above (6,000-dim spectra, a `hospital` column, species labels) and register its path in `configs/config.yaml`.

---

## 7. Configuration

Experiments are fully driven by **YAML config files** under [configs/](configs/). Two layers exist:

1. **Global data registry — [configs/config.yaml](configs/config.yaml)**: maps logical dataset names (`DRIAMS_FULL`, `MARISMa_FULL`, `RKI_FULL`, `MSUMG_FULL`, …) to file paths. Edit once for your machine.

2. **Per-experiment config** — e.g. [configs/vae_multidecoder_prior/vae_multidecoder_prior_driamsABC_marisma_rki.yaml](configs/vae_multidecoder_prior/vae_multidecoder_prior_driamsABC_marisma_rki.yaml):

```yaml
experiment:
  name: vae_multidecoder_prior
  seed: 42
  output_dir: experiments/results

data:
  domains: ["DRIAMS_A", "DRIAMS_B", "DRIAMS_C", "MARISMA", "RKI"]   # SOURCE domains
  species_list: ["Klebsiella_Pneumoniae", "Escherichia_Coli", "Staphylococcus_Aureus",
                 "Pseudomonas_Aeruginosa", "Enterococcus_Faecium", "Enterobacter_cloacae_complex"]
  normalization: row_minmax
  test_size: 0.2
  batch_size: 128
  use_species_weights: False

model:
  type: vae_multidecoder_prior     # selects the model class (see §3)
  latent_dim: 64
  num_domains: 5                   # must match number of training domains
  n_species: 6

training:
  epochs: 100
  lr: 1e-4
  annealing_epochs: 100
  patience: 20
  metrics_plot_title: "VAE Multidecoder w/prior for species"

evaluation:
  use_domain: false
  compute_tsne: true
  prior_sampling: false
  prior_samples_per_species: 50
```

For **AMR** experiments, use a config from [configs/vae_multidecoder_prior_multiamr/](configs/vae_multidecoder_prior_multiamr/) with `model.type: vae_multidecoder_prior_amr_head_z`, plus `model.n_antibiotics`, `training.lambda_amr`, and a `data.antibiotics_filter`.

---

## 8. How to Use It

You **do not need to train DALMA from scratch**. Two ready-to-use checkpoints ship with the repository, one per downstream setting:

| Use case | Checkpoint | Model class | Config |
|----------|-----------|-------------|--------|
| **Microbial identification** | `experiments/results/vae_multidecoder_prior/20260409_100009/model.pth` | `MultiVAE_Bernoulli_SpeciesPrior_Extended` | `latent_dim=64, num_domains=5, n_species=6` |
| **AMR prediction** | `experiments/results/vae_multidecoder_prior_multiamr_all_species_all_abs/20260716_150449/model.pth` | `MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ` | `latent_dim=128, num_domains=4, n_species=6, n_antibiotics=10` |

In almost all cases you should just load the checkpoint that matches your task, extract latent representations (and, for AMR, use the trained auxiliary head), and run the downstream tasks in §9. Training your own model is only needed if you want to change the architecture, add new source centers, or retrain a variant.

> The two checkpoints are trained with **different hyperparameters** — mind `latent_dim` (64 vs. 128) and `num_domains` (5 with RKI vs. 4 without). Instantiate the model exactly as shown below before loading the weights, or `load_state_dict` will fail.

### Recommended — Use the pre-trained DALMA encoder

At inference only the shared encoder is used. Load the checkpoint with the helpers in [src/evaluation/eval.py](src/evaluation/eval.py) and extract the posterior mean `z* = μ(x)`.

**For microbial identification:**

```python
import torch
from src.evaluation.eval import load_model, encode_latent
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "experiments/results/vae_multidecoder_prior/20260409_100009/model.pth"

model = load_model(
    MultiVAE_Bernoulli_SpeciesPrior_Extended(
        input_dim=6000, latent_dim=64, num_domains=5, n_species=6),
    CHECKPOINT,
)

Z = encode_latent(model, X_spectra, device)   # (N, 64) latent representations
```

**For AMR prediction:**

```python
import torch
from src.evaluation.eval import load_model, encode_latent
from models.deep.MultiVAEPriorAMRHeadZ import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "experiments/results/vae_multidecoder_prior_multiamr_all_species_all_abs/20260716_150449/model.pth"

model = load_model(
    MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ(
        input_dim=6000, latent_dim=128, num_domains=4, n_species=6, n_antibiotics=10),
    CHECKPOINT,
)

Z = encode_latent(model, X_spectra, device)   # (N, 128) latent representations
# The trained auxiliary AMR head (model.amr_heads) is used directly for resistance prediction.
```

`X_spectra` is an `(N, 6000)` array of preprocessed, row-minmax-normalized spectra (see [§6](#6-input-data-format)). Because the domain-specific decoders are discarded at inference, **spectra from unseen centers need no domain id** — they all pass through the same encoder. The resulting `Z` is the input to the downstream tasks in [§9](#9-downstream-tasks--experiments).

---

## 9. Downstream Tasks & Experiments

All experiment scripts auto-locate the project root and can be run from anywhere inside the repo.

### 9.1 Microbial Identification (zero-shot)

Freezes the encoder, fits a **linear probe** on source-domain latents, and evaluates on held-out target centers (DRIAMS-D, MS-UMG). Reports Balanced Accuracy, F1, recall, specificity, AUROC.

```bash
python experiments/species_id/run_classification_mlp.py --architecture vae_multidecoder_prior
```

Available `--architecture` choices: `vae`, `vae_prior`, `vae_multidecoder`, `vae_multidecoder_prior` (DALMA), `dann`, `coral`. For the Maldi Transformer baseline use [run_classification_mlp_maldi_transformer.py](experiments/species_id/run_classification_mlp_maldi_transformer.py).

### 9.2 AMR Prediction (zero-shot & few-shot)

Uses the auxiliary AMR head. The paper evaluates *Klebsiella pneumoniae* across five clinically relevant antibiotics (Imipenem, Meropenem, Ceftazidime, Ciprofloxacin, Piperacillin-Tazobactam), the ones with confirmed proteomic biomarkers in the MALDI-TOF detection range.

```bash
python experiments/amr/run_classification_mlp.py
```

Few-shot adaptation grids (progressively adding labeled target isolates) live under [experiments/finetuning/](experiments/finetuning/).

#### Predicting AMR for a species–antibiotic pair

The shipped AMR checkpoint has **one head per antibiotic** (`model.amr_heads[j]`), each taking the latent `z* = μ(x)` and producing a single resistance logit. The head index `j` follows the training `antibiotics_filter` order:

| j | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Antibiotic | Imipenem | Meropenem | Ceftazidime | Ciprofloxacin | Piperacillin-Tazobactam | Amikacin | Oxacillin | Clindamycin | Erythromycin | Vancomycin |

For *K. pneumoniae*, the first five (indices 0–4) are the clinically relevant ones reported in the paper. To predict resistance for a given species–antibiotic pair, feed spectra of that species and read the corresponding head:

```python
import numpy as np
import torch
from src.evaluation.eval import load_model
from models.deep.MultiVAEPriorAMRHeadZ import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANTIBIOTICS = ["Imipenem", "Meropenem", "Ceftazidime", "Ciprofloxacin",
               "Piperacillin-Tazobactam", "Amikacin", "Oxacillin",
               "Clindamycin", "Erythromycin", "Vancomycin"]

CHECKPOINT = "experiments/results/vae_multidecoder_prior_multiamr_all_species_all_abs/20260716_150449/model.pth"
model = load_model(
    MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ(
        input_dim=6000, latent_dim=128, num_domains=4, n_species=6, n_antibiotics=10),
    CHECKPOINT,
)
model.eval().to(device)

# X_kpn: (N, 6000) Klebsiella pneumoniae spectra, preprocessed + row-minmax normalized
antibiotic = "Meropenem"
j = ANTIBIOTICS.index(antibiotic)

with torch.no_grad():
    x = torch.tensor(X_kpn, dtype=torch.float32, device=device)
    mu, _ = model.encoder(x)                    # z* = μ(x)
    logit = model.amr_heads[j](mu).squeeze(1)   # head for antibiotic j
    prob_resistant = torch.sigmoid(logit).cpu().numpy()

pred = (prob_resistant >= 0.5).astype(int)      # 1 = resistant, 0 = susceptible
```

If you already have ground-truth labels and just want the metrics (AUROC, PR-AUC, balanced accuracy), use the ready-made helper — it reports every antibiotic with ≥10 labeled samples and both classes present:

```python
from src.evaluation.eval import evaluate_amr_head

# amr_labels: (N, 10) array with {0, 1, NaN}
results = evaluate_amr_head(model, X_kpn, amr_labels, ANTIBIOTICS, device)
print(results["Meropenem"])   # {'auc': ..., 'bal_acc': ..., 'pr_auc': ..., 'n': ...}
```

### 9.3 Novelty / OOD Detection (selective prediction)

Fits a **GMM** on source-domain latents and uses its likelihood as a novelty score. Rejecting low-likelihood spectra improves Balanced Accuracy at the cost of coverage.

```bash
python experiments/novelty_detection/combined_detector.py
```

### 9.4 Domain Similarity Analysis

Quantifies domain shift by comparing normalized pairwise cosine similarity between centers in the raw vs. DALMA latent space.

```bash
python experiments/spectra_similarity/compute_similarity.py
```

---

## 10. Benchmark Datasets

The benchmark comprises **seven acquisition domains** from four public repositories, across three countries:

| Dataset | Country | Institution | Years | Instrument |
|---------|---------|-------------|-------|------------|
| DRIAMS-A | Switzerland | Univ. Hospital Basel | 2015–2018 | Microflex LT/SH + Smart LS |
| DRIAMS-B | Switzerland | Canton Hospital Basel-Land | 2018 | Microflex LT-SH |
| DRIAMS-C | Switzerland | Canton Hospital Aarau | 2018 | Microflex LT-SH |
| DRIAMS-D | Switzerland | Viollier AG | 2018 | Microflex Smart LS |
| MARISMa | Spain | H.G.U. Gregorio Marañón | 2018–2024 | Microflex LT/SH + Smart LS |
| RKI | Germany | Robert Koch Institute | various | Autoflex |
| MS-UMG | Germany | Univ. Medical Center Göttingen | 2020–2021 | Microflex LT-SH + Smart |

**Source–target protocol (zero-shot):** models are trained on **source** domains (DRIAMS-A/B/C, MARISMa, RKI) and evaluated directly on **held-out targets** (DRIAMS-D — moderate shift; MS-UMG — severe shift: different instrument, geography, period, and preprocessing) with **no adaptation**.

Six clinically relevant species are used: *Enterobacter cloacae* complex (ECC), *Escherichia coli*, *Enterococcus faecium*, *Klebsiella pneumoniae*, *Pseudomonas aeruginosa*, and *Staphylococcus aureus*. AMR evaluation focuses on *K. pneumoniae*.

The datasets are publicly available from their original sources (DRIAMS, MARISMa, RKI, MS-UMG); this repository does not redistribute them.

---

## 11. Citation

If you use this code or DALMA in your research, please cite:

```bibtex
@misc{garcíanavarro2026biologicallyinformedrepresentationlearning,
      title={Biologically Informed Representation Learning for Robust Cross-Center Generalization of MALDI-TOF Mass Spectrometry}, 
      author={Alejandro L. García-Navarro and Carlos Sevilla-Salcedo and Belén Rodríguez-Sánchez and Vanessa Gómez-Verdejo},
      year={2026},
      eprint={2608.08182},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.08182}, 
}
```

**Corresponding author:** Alejandro L. García-Navarro — `agnavarr@pa.uc3m.es`
Department of Signal Theory and Communications, Universidad Carlos III de Madrid, and Instituto de Investigación Sanitaria Gregorio Marañón (IiSGM).