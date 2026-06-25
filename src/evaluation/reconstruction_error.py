import torch
import numpy as np
from scipy.stats import norm


def reconstruction_error(model, X, domain_id, device, batch_size=256):
    """
    Reconstruye X con el decoder indicado y devuelve MSE por muestra.
    """
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            d_batch  = torch.full((len(x_batch),), domain_id, dtype=torch.long).to(device)
            mu, _    = model.encoder(x_batch)
            x_recon  = model.decoder(mu, d_batch)
            err      = ((x_batch - x_recon) ** 2).mean(dim=1)
            errors.append(err.cpu().numpy())
    return np.concatenate(errors)


def recon_ood_score(errors, mu_log, sig_log):
    """
    Score OOD basado en la log-normal ajustada.
    Score alto = error improbable bajo esa log-normal = OOD.
    """
    log_errs = np.log(errors + 1e-10)
    return -norm.logpdf(log_errs, loc=mu_log, scale=sig_log)
