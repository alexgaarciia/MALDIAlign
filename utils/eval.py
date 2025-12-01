import torch
import numpy as np

def eval_model(model, dataloader, device, use_domain=False):
    """
    Compute latent means (mus) for all samples in the dataloader.
    Works for both standard and multi-domain VAEs.
    """
    model.eval()
    model.to(device)
    mus_all = []

    with torch.no_grad():
        for batch in dataloader:
            if use_domain:
                x, domain = batch
                x, domain = x.to(device), domain.to(device)
                mu, logvar = model.encoder(x, domain)
            else:
                x, _ = batch
                x = x.to(device)
                mu, logvar = model.encoder(x)

            mus_all.append(mu.cpu().numpy())

    mus_all = np.concatenate(mus_all, axis=0)
    return mus_all
