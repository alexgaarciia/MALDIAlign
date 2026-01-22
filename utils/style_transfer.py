import torch
import numpy as np


def style_transfer(model, X, domain_target_id, device, batch_size=256, deterministic=True):
    """
    Perform domain style transfer using a trained conditional generative model.

    This function maps input samples from their original domain into the
    *style* of a target domain by encoding them into a latent space and
    decoding them while conditioning on a specified target domain ID.

    The typical use case is domain alignment, where spectral samples
    (e.g. MALDI-TOF or FTIR) are transferred to a reference domain
    (hospital, batch, instrument, etc.) before downstream classification.

    Parameters
    ----------
    model : torch.nn.Module
        Trained conditional generative model with:
        - `encoder(x)` → (mu, logvar)
        - `decoder(z, domain_id)` → x_hat
        Optionally:
        - `reparameterize(mu, logvar)` if `deterministic=False`.

    X : np.ndarray of shape (n_samples, n_features)
        Input data matrix containing the original samples to be transferred.
        Each row corresponds to one spectrum/sample.

    domain_target_id : int
        Integer identifier of the target domain to which all samples
        will be transferred.

    device : torch.device
        Device on which the computation will be performed
        (e.g. `torch.device("cuda")` or `torch.device("cpu")`).

    batch_size : int, optional (default=256)
        Number of samples processed per batch during inference.

    deterministic : bool, optional (default=True)
        If True, uses the latent mean `mu` as the latent representation
        (deterministic style transfer).
        If False, samples from the latent distribution using the
        reparameterization trick (`z ~ N(mu, exp(logvar))`).

    Returns
    -------
    X_transferred : np.ndarray of shape (n_samples, n_features)
        Output data matrix where each input sample has been transferred
        to the target domain style.
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    loader = torch.utils.data.DataLoader(
        X_tensor, batch_size=batch_size, shuffle=False
    )

    X_out = []

    with torch.no_grad():
        for x in loader:
            x = x.to(device) 

            mu, logvar = model.encoder(x) 
            z = mu if deterministic else model.reparameterize(mu, logvar)

            domain_id = torch.full(
                (x.size(0),),
                domain_target_id,
                dtype=torch.long,
                device=device
            ) 

            x_hat = model.decoder(z, domain_id)  
            X_out.append(x_hat.cpu().numpy())

    return np.vstack(X_out)
