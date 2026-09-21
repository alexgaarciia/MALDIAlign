from models.deep.VAEBernoulli import VAE_Bernoulli_Extended
from models.deep.MultiVAE import MultiVAE_Bernoulli_Extended
from models.deep.MultiVAECoral import MultiVAE_CORAL
from models.deep.cVAE import ConditionalVAE_Bernoulli_Extended
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended
from models.deep.DANN import DANNFull_Extended
from models.deep.MultiVAEPriorAMRHeadZ import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ
from models.deep.MultiVAEPriorAMRHeadZEmb import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZEmb
from models.deep.MultiVAEPriorAdv import MultiVAE_Bernoulli_SpeciesPrior_Adv_Extended


def build_model(cfg: dict, input_dim: int, antibiotic_names=None, num_domains=None):
    """
    Instantiate a model based on the configuration dictionary.

    This function acts as a factory that maps the `model` section of the YAML
    configuration to a concrete PyTorch model class. Due to the current design
    of the model classes, training-related parameters are also passed to the
    model constructor for compatibility.

    Parameters
    ----------
    cfg : dict
        Full experiment configuration dictionary loaded from YAML. Must contain
        at least the keys:
        - cfg["model"]
        - cfg["training"]
    input_dim : int
        Dimensionality of the input data (number of features per sample).

    Returns
    -------
    model : torch.nn.Module
        Instantiated (but untrained) model.
    """

    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    model_type = model_cfg["type"]

    if num_domains is not None and "num_domains" not in model_cfg:
        model_cfg = {**model_cfg, "num_domains": num_domains}

    if model_type == "vae_bernoulli":
        model = VAE_Bernoulli_Extended(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            num_domains=model_cfg.get("num_domains", 1),
            use_species_prior=model_cfg.get("use_species_prior", False),
            n_species=model_cfg.get("n_species", None),
            epochs=training_cfg["epochs"],
            annealing_epochs=training_cfg["annealing_epochs"],
            patience=training_cfg["patience"])

    elif model_type == "vae_multidecoder":
        model = MultiVAE_Bernoulli_Extended(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            num_domains=model_cfg["num_domains"],
            epochs=training_cfg["epochs"],
            annealing_epochs=training_cfg["annealing_epochs"],
            patience=training_cfg["patience"]
        )

    elif model_type == "vae_multidecoder_coral":
        model = MultiVAE_CORAL(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            num_domains=model_cfg["num_domains"],
            epochs=training_cfg["epochs"],
            annealing_epochs=training_cfg["annealing_epochs"],
            patience=training_cfg["patience"],
        )

    elif model_type == "cvae":
        model = ConditionalVAE_Bernoulli_Extended(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            num_domains=model_cfg["cond_dim"],
            epochs=training_cfg["epochs"],
            annealing_epochs=training_cfg["annealing_epochs"],
            patience=training_cfg["patience"],
        )

    elif model_type == "vae_multidecoder_prior":
        model = MultiVAE_Bernoulli_SpeciesPrior_Extended(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            num_domains=model_cfg["num_domains"],
            n_species=model_cfg["n_species"],
            epochs=training_cfg["epochs"],
            annealing_epochs=training_cfg["annealing_epochs"],
            patience=training_cfg["patience"],
        )

    elif model_type == "dann":
        model = DANNFull_Extended(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            n_species=model_cfg["n_species"],
            n_domains=model_cfg["num_domains"],
            epochs=training_cfg["epochs"],
            lr=float(training_cfg["lr"]),
            patience=training_cfg["patience"],
        )

    elif model_type == "vae_multidecoder_prior_amr_head_z":
        model = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            num_domains=model_cfg["num_domains"],
            n_species=model_cfg["n_species"],
            n_antibiotics=model_cfg["n_antibiotics"],
            lambda_amr=training_cfg["lambda_amr"],
            antibiotic_names=antibiotic_names,
            epochs=training_cfg["epochs"],
            annealing_epochs=training_cfg.get("annealing_epochs", None),
            patience=training_cfg["patience"],
            use_fixed_prior = model_cfg.get("use_fixed_prior", False),
        )
    
    elif model_type == "vae_multidecoder_prior_amr_head_zemb":
        model = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZEmb(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            num_domains=model_cfg["num_domains"],
            n_species=model_cfg["n_species"],
            n_antibiotics=model_cfg["n_antibiotics"],
            lambda_amr=training_cfg["lambda_amr"],
            antibiotic_names=antibiotic_names,
            epochs=training_cfg["epochs"],
            annealing_epochs=training_cfg.get("annealing_epochs", None),
            patience=training_cfg["patience"],
            use_fixed_prior=model_cfg.get("use_fixed_prior", False),
            species_emb_dim=model_cfg.get("species_emb_dim", 128),
        )

    elif model_type == "vae_multidecoder_prior_adv":
        model = MultiVAE_Bernoulli_SpeciesPrior_Adv_Extended(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            num_domains=model_cfg["num_domains"],
            n_species=model_cfg["n_species"],
            epochs=training_cfg["epochs"],
            annealing_epochs=training_cfg.get("annealing_epochs", 50),
            patience=training_cfg["patience"],
            adv_eps=model_cfg.get("adv_eps", 0.02),
            adv_lambda=model_cfg.get("adv_lambda", 0.5),
            adv_attack=model_cfg.get("adv_attack", "fgsm"),
            pgd_steps=model_cfg.get("pgd_steps", 7),
            pgd_alpha=model_cfg.get("pgd_alpha", None),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model
