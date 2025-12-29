import torch
import torch.nn as nn
from models.deep.networks import EncoderDANN, SpeciesClassifier, DomainClassifier, grad_reverse


class DANN(nn.Module):
    """
    Baseline (non-adversarial) DANN-style model.

    Consists of:
    - A shared feature encoder
    - A species classifier head

    This class is useful as:
    - A strong supervised baseline on the latent space
    - A warm-start model for initializing DANNFull
    - A way to evaluate the quality of the encoder without domain adversariality

    IMPORTANT:
    This model DOES NOT include domain adversarial training.
    """

    def __init__(self, input_dim: int, latent_dim: int, n_species: int):
        super().__init__()

        self.encoder = EncoderDANN(input_dim, latent_dim)
        self.classifier = SpeciesClassifier(latent_dim, n_species)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data (batch_size, input_dim)

        Returns
        -------
        z : torch.Tensor
            Latent representation (batch_size, latent_dim)
        species_logits : torch.Tensor
            Species classification logits
        """
        z = self.encoder(x)
        species_logits = self.classifier(z)
        return z, species_logits
    

class DANNFull(nn.Module):
    """
    Full Domain-Adversarial Neural Network (DANN).

    Architecture:
    - Shared encoder
    - Species classifier head (task objective)
    - Domain classifier head (adversarial objective via GRL)

    Training objective:
        min_{encoder, species_clf} L_species
        max_{encoder}              L_domain
        min_{domain_clf}           L_domain

    This is achieved by the Gradient Reversal Layer (GRL).

    Notes
    -----
    - lambda_ controls the strength of domain adversarial pressure.
    - Setting lambda_=0 reduces this model to a standard multi-head classifier.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_species: int,
        n_domains: int = 2,
    ):
        super().__init__()

        # Shared feature extractor
        self.encoder = EncoderDANN(input_dim, latent_dim)

        # Task-specific heads
        self.species_clf = SpeciesClassifier(latent_dim, n_species)
        self.domain_clf = DomainClassifier(latent_dim, n_domains)

    def forward(self, x: torch.Tensor, lambda_: float = 0.0):
        """
        Forward pass through the full DANN model.

        Parameters
        ----------
        x : torch.Tensor
            Input data (batch_size, input_dim)
        lambda_ : float
            Gradient reversal strength.
            - 0.0  → no domain adversarial effect
            - >0.0 → increasing domain confusion pressure

        Returns
        -------
        species_logits : torch.Tensor
            Logits for species classification
        domain_logits : torch.Tensor
            Logits for domain classification
        """
        # Shared representation
        z = self.encoder(x)

        # Species prediction (normal gradient flow)
        species_logits = self.species_clf(z)

        # Domain prediction (reversed gradient flow)
        z_rev = grad_reverse(z, lambda_)
        domain_logits = self.domain_clf(z_rev)

        return species_logits, domain_logits
