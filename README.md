# MALDIAlign

**Biologically Informed Representation Learning for Robust Cross-Center Generalization of MALDI-TOF Mass Spectrometry**

Official implementation of **DALMA** (*Domain ALignment for MALDI-TOF MS*), a probabilistic representation learning framework that maps MALDI-TOF mass spectra to a latent space that keeps biological information while being robust to center-specific variability. This allows **zero-shot deployment on previously unseen centers**.

> García-Navarro, Sevilla-Salcedo, Rodríguez-Sánchez, Gómez-Verdejo. *Biologically Informed Representation Learning for Robust Cross-Center Generalization of MALDI-TOF Mass Spectrometry* (under review).

---

## What DALMA does

- A **shared encoder** `q(z|x)` maps spectra from all centers into a common latent space.
- **Domain-specific decoders** (one per training center) absorb acquisition-specific variability during training.
- A **species-conditioned prior** `p(z|s)` pulls spectra of the same species together across centers.
- An optional **auxiliary AMR head** injects resistance information for antimicrobial resistance prediction.

Domain information is only used during training. At inference only the shared encoder is used (`z = μ(x)`), so spectra from any center, seen or unseen, go through the same model.

## Installation

```bash
git clone https://github.com/alexgaarciia/MALDIAlign.git
cd MALDIAlign
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and PyTorch 2.x. [requirements.txt](requirements.txt) targets CUDA 12.8; for CPU or another CUDA version, install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/) first.

## Input data

DALMA works on **preprocessed, binned spectra**: an `(N, 6000)` array (2,000–20,000 m/z, 3 Da bins, log-scaled), min-max normalized per spectrum so values lie in `[0, 1]`. Datasets are not redistributed; get them from their original sources (DRIAMS, MARISMa, RKI, MS-UMG).

## Using the pre-trained models

Two checkpoints are included, one per task. Instantiate the model with the exact hyperparameters below before loading the weights.

| Task | Checkpoint | Model | Hyperparameters |
|------|-----------|-------|-----------------|
| Microbial identification | `experiments/results/vae_multidecoder_prior/20260409_100009/model.pth` | `MultiVAE_Bernoulli_SpeciesPrior_Extended` | `latent_dim=64, num_domains=5, n_species=6` |
| AMR prediction | `experiments/results/vae_multidecoder_prior_multiamr_all_species_all_abs/20260716_150449/model.pth` | `MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ` | `latent_dim=128, num_domains=4, n_species=6, n_antibiotics=10` |

### Latent representations (species identification)

```python
import torch
from src.evaluation.eval import load_model, encode_latent
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(
    MultiVAE_Bernoulli_SpeciesPrior_Extended(
        input_dim=6000, latent_dim=64, num_domains=5, n_species=6),
    "experiments/results/vae_multidecoder_prior/20260409_100009/model.pth",
)

Z = encode_latent(model, X_spectra, device) 
```

`Z` can then be used as input to any classifier (e.g. a linear probe) trained on source centers and applied directly to new ones.

### AMR prediction

The AMR model has one head per antibiotic (`model.amr_heads[j]`), applied on the latent `z = μ(x)`. Head order:

`Imipenem, Meropenem, Ceftazidime, Ciprofloxacin, Piperacillin-Tazobactam, Amikacin, Oxacillin, Clindamycin, Erythromycin, Vancomycin`

```python
import torch
from src.evaluation.eval import load_model
from models.deep.MultiVAEPriorAMRHeadZ import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANTIBIOTICS = ["Imipenem", "Meropenem", "Ceftazidime", "Ciprofloxacin",
               "Piperacillin-Tazobactam", "Amikacin", "Oxacillin",
               "Clindamycin", "Erythromycin", "Vancomycin"]

model = load_model(
    MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ(
        input_dim=6000, latent_dim=128, num_domains=4, n_species=6, n_antibiotics=10),
    "experiments/results/vae_multidecoder_prior_multiamr_all_species_all_abs/20260716_150449/model.pth",
)
model.eval().to(device)

j = ANTIBIOTICS.index("Meropenem")
with torch.no_grad():
    x = torch.tensor(X_spectra, dtype=torch.float32, device=device)
    mu, _ = model.encoder(x)
    prob_resistant = torch.sigmoid(model.amr_heads[j](mu).squeeze(1)).cpu().numpy()
```

If you have ground-truth labels, `evaluate_amr_head` in [src/evaluation/eval.py](src/evaluation/eval.py) reports AUROC, PR-AUC and balanced accuracy per antibiotic.

## Repository structure

```
models/        DALMA, ablations and baselines (VAE, DANN, CORAL, cVAE, MLP)
src/           data loading, training, evaluation, fine-tuning, visualization
experiments/   training entrypoint (run_experiment.py) and downstream experiment scripts
```

Training your own model is only needed to change the architecture or add source centers. It is done with `python experiments/run_experiment.py --config <your_config>.yaml`, where the config is a YAML file with `experiment`, `data`, `model`, `training` and `evaluation` sections (see [experiments/run_experiment.py](experiments/run_experiment.py)). 

## Citation

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

**Contact:** Alejandro L. García-Navarro, `agnavarr@pa.uc3m.es`. Instituto de Investigación Sanitaria Gregorio Marañón (IiSGM), and Department of Signal Theory and Communications, Universidad Carlos III de Madrid.
