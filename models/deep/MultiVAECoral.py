import torch
import torch.optim as optim
from models.deep.MultiVAE import MultiVAE_Bernoulli


class MultiVAE_CORAL(MultiVAE_Bernoulli):
    """
    Multi-decoder VAE with Multi-domain CORAL regularization.
    Aligns second-order statistics across all domains present in the training batch.
    """

    def __init__(self, input_dim, latent_dim, num_domains, epochs=100, lr=1e-4, annealing_epochs=50, patience=20, lambda_coral=1e-3):
        super().__init__(input_dim, latent_dim, num_domains)
        self.epochs = epochs
        self.lr = lr
        self.annealing_epochs = annealing_epochs
        self.patience = patience
        self.lambda_coral = lambda_coral

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

        self.loss_during_training = []
        self.reconstruc_during_training = []
        self.KL_during_training = []
        self.coral_loss_during_training = []

    def compute_covariance(self, z):
        """Computes the covariance matrix of a set of latent vectors."""
        n = z.size(0)
        if n <= 1:
            return None
        
        z_centered = z - z.mean(0, keepdim=True)
        cov = (z_centered.T @ z_centered) / (n - 1)
        return cov

    def compute_coral_loss(self, cov_a, cov_b):
        """Computes Frobenius norm distance between two covariance matrices."""
        d = cov_a.size(0)
        loss = torch.mean((cov_a - cov_b) ** 2) 
        return loss / (4 * (d ** 2))

    def trainloop(self, trainloader, validloader, device):
        self.to(device)
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            beta = 1

            # =======================
            #        TRAIN
            # =======================
            self.train()
            tr_total, tr_recon, tr_kl, tr_coral = 0.0, 0.0, 0.0, 0.0

            for batch in trainloader:
                x, domain_id = batch[0].to(device), batch[1].to(device)
                self.optimizer.zero_grad()

                mu, logvar, z = self.forward(x, domain_id)
                loss, recon, kl = self.elbo_loss(x, mu, logvar, z, domain_id, beta)

                unique_domains = domain_id.unique()
                coral_val = torch.tensor(0.0).to(device)
                
                if len(unique_domains) > 1:
                    domain_covs = {}
                    for d in unique_domains:
                        z_d = z[domain_id == d]
                        cov_d = self.compute_covariance(z_d)
                        if cov_d is not None:
                            domain_covs[d.item()] = cov_d

                    pairs = 0
                    ids = list(domain_covs.keys())
                    for i in range(len(ids)):
                        for j in range(i + 1, len(ids)):
                            coral_val += self.compute_coral_loss(domain_covs[ids[i]], domain_covs[ids[j]])
                            pairs += 1
                    
                    if pairs > 0:
                        coral_val = coral_val / pairs
                        loss = loss + self.lambda_coral * coral_val

                loss.backward()
                self.optimizer.step()

                tr_total += loss.item()
                tr_recon += recon.item()
                tr_kl += kl.item()
                tr_coral += coral_val.item()

            # =======================
            #       VALIDATION
            # =======================
            self.eval()
            val_total, val_recon, val_kl, val_coral = 0.0, 0.0, 0.0, 0.0
            with torch.no_grad():
                for batch in validloader:
                    x, domain_id = batch[0].to(device), batch[1].to(device)
                    mu, logvar, z = self.forward(x, domain_id)
                    loss, recon, kl = self.elbo_loss(x, mu, logvar, z, domain_id, beta)
                    
                    unique_domains = domain_id.unique()
                    coral_val = torch.tensor(0.0).to(device)
                    if len(unique_domains) > 1:
                        domain_covs = {d.item(): self.compute_covariance(z[domain_id == d]) 
                                       for d in unique_domains if z[domain_id == d].size(0) > 1}
                        ids = list(domain_covs.keys())
                        pairs = 0
                        for i in range(len(ids)):
                            for j in range(i + 1, len(ids)):
                                if domain_covs[ids[i]] is not None and domain_covs[ids[j]] is not None:
                                    coral_val += self.compute_coral_loss(domain_covs[ids[i]], domain_covs[ids[j]])
                                    pairs += 1
                        if pairs > 0:
                            coral_val = coral_val / pairs
                            loss = loss + self.lambda_coral * coral_val

                    val_total += loss.item()
                    val_recon += recon.item()
                    val_kl += kl.item()
                    val_coral += coral_val.item()

            n_tr, n_val = len(trainloader), len(validloader)
            self.loss_during_training.append((tr_total/n_tr, val_total/n_val))
            self.coral_loss_during_training.append((tr_coral/n_tr, val_coral/n_val))
            self.reconstruc_during_training.append((tr_recon/n_tr, val_recon/n_val))
            self.KL_during_training.append((tr_kl/n_tr, val_kl/n_val))

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:03d} | [Train] Loss: {tr_total/n_tr:.4f} | CORAL: {tr_coral/n_tr:.6f} || "
                      f"[Val] Loss: {val_total/n_val:.4f} | CORAL: {val_coral/n_val:.6f}")

            current_val_loss = val_total / n_val
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.load_state_dict(best_state)
