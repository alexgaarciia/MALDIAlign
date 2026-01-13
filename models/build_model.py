from models.deep.VAEBernoulli import VAE_Bernoulli_Extended
from models.deep.MultiVAE import MultiVAE_Bernoulli_Extended
from models.deep.MultiVAECoral import MultiVAE_CORAL
from models.deep.cVAE import ConditionalVAE_Bernoulli_Extended
from models.deep.cVAEInvariant import InvariantCVAE_Bernoulli_Extended
from models.deep.cVAEInvariantSpecies import InvariantCVAESpecies_Bernoulli_Extended
from models.deep.cVAESpecies import SpeciesCVAE_Bernoulli_Extended
from models.deep.cVAEPrior import ConditionalVAE_Bernoulli_SpeciesPrior_Extended
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended
from models.deep.DANN import DANNFull_Extended


def build_model(cfg: dict, input_dim: int):
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

    if model_type == "vae_bernoulli":
        model = VAE_Bernoulli_Extended(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            epochs=training_cfg["epochs"],
            annealing_epochs=training_cfg["annealing_epochs"],
            patience=training_cfg["patience"],
        )

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
            cond_dim=model_cfg["cond_dim"],
            epochs=training_cfg["epochs"],
            annealing_epochs=training_cfg["annealing_epochs"],
            patience=training_cfg["patience"],
        )

    elif model_type == "cvae_invariant":
        model = InvariantCVAE_Bernoulli_Extended(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            cond_dim=model_cfg["cond_dim"],
            epochs=training_cfg["epochs"],
            annealing_epochs=training_cfg["annealing_epochs"],
            patience=training_cfg["patience"],
        )

    elif model_type == "cvae_invariant_species":
        model =  InvariantCVAESpecies_Bernoulli_Extended(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            n_species=model_cfg["cond_dim"],
            epochs=training_cfg["epochs"],
            annealing_epochs=training_cfg["annealing_epochs"],
            patience=training_cfg["patience"],
        )

    elif model_type == "cvae_species":
        model =  SpeciesCVAE_Bernoulli_Extended(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            n_species=model_cfg["cond_dim"],
            epochs=training_cfg["epochs"],
            annealing_epochs=training_cfg["annealing_epochs"],
            patience=training_cfg["patience"],
        )

    elif model_type == "cvae_species_prior":
        model = ConditionalVAE_Bernoulli_SpeciesPrior_Extended(
            input_dim=input_dim,
            latent_dim=model_cfg["latent_dim"],
            cond_dim=model_cfg["cond_dim"],      
            n_species=model_cfg["n_species"],   
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
            n_domains=model_cfg.get("num_domains", 2),
            source_domain_id=model_cfg.get("source_domain_id", 0),
            epochs=training_cfg["epochs"],
            lr=float(training_cfg["lr"]),
            lambda_domain=float(model_cfg.get("lambda_domain", 0.01)),
            patience=training_cfg["patience"],
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model
