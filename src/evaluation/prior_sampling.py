import torch


def sample_prior(model, species_id, n_samples, device):
    """
    Sample latent vectors from the species-conditioned prior.

    Parameters
    ----------
    model : torch.nn.Module
        Trained VAE model with ConditionalPrior.
    species_id : int
        Species index to sample from.
    n_samples : int
        Number of latent samples.
    device : torch.device
        Device.

    Returns
    -------
    z : torch.Tensor
        Sampled latent vectors (n_samples, latent_dim).
    """

    species = torch.full((n_samples,), species_id, dtype=torch.long).to(device)

    mu = model.prior.mu_embed(species)
    logvar = model.prior.logvar_embed(species)

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    z = mu + eps * std

    return z


def sample_all_species_priors(model, n_species, n_samples, device):
    """
    Sample latent points from priors of all species.
    """

    all_samples = []
    labels = []

    for s in range(n_species):

        z = sample_prior(model, s, n_samples, device)

        all_samples.append(z.detach().cpu())
        labels.extend([s] * n_samples)

    Z = torch.cat(all_samples, dim=0)

    return Z, labels
