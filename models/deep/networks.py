import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ============================================================
#                       ENCODERS
# ============================================================

class Encoder(nn.Module):
    """
    Standard encoder for a Variational Autoencoder (VAE).

    Maps an input spectrum x ∈ R^{input_dim} to the parameters of a Gaussian
    latent distribution q(z|x) = N(μ(x), σ²(x)).

    Notes
    -----
    - The output is (mu, logvar), not a sampled z.
    - logvar is clamped for numerical stability.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns
        -------
        mu : torch.Tensor
            Mean of the approximate posterior q(z|x)
        logvar : torch.Tensor
            Log-variance of the approximate posterior q(z|x)
        """
        h = self.net(x)
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -6, 6)
        return mu, logvar


class ConditionalEncoder(nn.Module):
    """
    Conditional VAE encoder.

    Encodes x conditioned on auxiliary information c (e.g. species, domain).

    q(z | x, c) = N(μ(x,c), σ²(x,c))
    """

    def __init__(self, input_dim: int, latent_dim: int, cond_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data (batch_size, input_dim)
        c : torch.Tensor
            Conditioning vector (batch_size, cond_dim)

        Returns
        -------
        mu, logvar : torch.Tensor
            Parameters of q(z | x, c)
        """
        h = torch.cat([x, c], dim=1)
        hidden = self.net(h)
        mu = self.mu(hidden)
        logvar = torch.clamp(self.logvar(hidden), -6, 6)
        return mu, logvar


# ============================================================
#                       DECODERS
# ============================================================

class BernoulliDecoder(nn.Module):
    """
    Bernoulli decoder for VAE reconstruction.

    Supports:
    - Single decoder (num_domains = 1)
    - Multiple domain-specific decoders (num_domains > 1)

    Reconstruction likelihood:
        p(x | z) = Bernoulli(θ(z))
    """

    def __init__(self, latent_dim: int, output_dim: int, num_domains: int = 1):
        super().__init__()
        self.num_domains = num_domains

        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Linear(1024, output_dim),
                nn.Sigmoid()
            )
            for _ in range(num_domains)
        ])

    def forward(self, z: torch.Tensor, domain_id: torch.Tensor | None = None):
        """
        Decode latent variable z into reconstructed input.

        Parameters
        ----------
        z : torch.Tensor
            Latent samples (batch_size, latent_dim)
        domain_id : torch.Tensor, optional
            Domain index per sample (required if num_domains > 1)

        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed input (batch_size, output_dim)
        """
        if self.num_domains == 1:
            return self.net[0](z)

        if domain_id is None:
            raise ValueError("domain_id must be provided when num_domains > 1")

        x_recon = torch.zeros(
            z.size(0),
            self.net[0][-2].out_features,
            device=z.device
        )

        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.any():
                x_recon[mask] = self.net[d](z[mask])

        return x_recon

    def log_prob(self, x: torch.Tensor, z: torch.Tensor, domain_id: torch.Tensor | None = None):
        """
        Compute log p(x | z) under a Bernoulli likelihood.

        Assumes x ∈ [0, 1].

        Returns
        -------
        log_prob : torch.Tensor
            Log-likelihood per sample (batch_size,)
        """
        theta = self.forward(z, domain_id)

        if torch.any(theta < 0) or torch.any(theta > 1) or torch.isnan(theta).any():
            raise ValueError("Decoder output out of bounds")

        if torch.any(x < 0) or torch.any(x > 1) or torch.isnan(x).any():
            raise ValueError("Input x must be in [0,1]")

        return -F.binary_cross_entropy(theta, x, reduction="none").sum(dim=1)


class ConditionalDecoder(nn.Module):
    """
    Conditional Bernoulli decoder p(x | z, c).
    """

    def __init__(self, latent_dim: int, output_dim: int, cond_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor):
        h = torch.cat([z, c], dim=1)
        return self.net(h)

    def log_prob(self, x: torch.Tensor, z: torch.Tensor, c: torch.Tensor):
        theta = self.forward(z, c)

        if torch.any(theta < 0) or torch.any(theta > 1):
            raise ValueError("Theta out of bounds")

        return -F.binary_cross_entropy(theta, x, reduction="none").sum(dim=1)


# ============================================================
#                 DANN-SPECIFIC MODULES
# ============================================================

class EncoderDANN(nn.Module):
    """
    Feature encoder used in Domain-Adversarial Neural Networks (DANN).

    Produces a deterministic latent representation z used by:
    - Species classifier
    - Domain discriminator (via Gradient Reversal Layer)
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class SpeciesClassifier(nn.Module):
    """
    Linear classifier for species prediction from latent space.
    """

    def __init__(self, latent_dim: int, n_species: int):
        super().__init__()
        self.classifier = nn.Linear(latent_dim, n_species)

    def forward(self, z: torch.Tensor):
        return self.classifier(z)


class DomainClassifier(nn.Module):
    """
    Domain discriminator used in DANN.
    """

    def __init__(self, latent_dim: int, n_domains: int = 2):
        super().__init__()
        self.classifier = nn.Linear(latent_dim, n_domains)

    def forward(self, z: torch.Tensor):
        return self.classifier(z)
    
    
class GradientReversal(Function):
    """
    Gradient Reversal Layer (GRL).

    Forward: identity
    Backward: multiplies gradient by -λ

    Used to enforce domain invariance in DANN.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float):
    """
    Apply Gradient Reversal Layer.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    lambda_ : float
        Gradient reversal strength

    Returns
    -------
    torch.Tensor
        Identity in forward pass, reversed gradients in backward
    """
    return GradientReversal.apply(x, lambda_)
