import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from models.deep.networks import ConditionalPrior
from models.deep.MultiVAE import MultiVAE_Bernoulli


class MultiVAE_Bernoulli_SpeciesPrior_Adv(MultiVAE_Bernoulli):
    """
    Multi-decoder VAE with species-conditional latent prior and adversarial training.

    During training, input spectra are perturbed using FGSM to improve robustness
    to small perturbations in MALDI-TOF spectra (instrument variation, calibration,
    sample preparation differences).

    Encoder: q(z | x)
    Prior:   p(z | species)
    Decoder: p(x | z, domain)
    """

    def __init__(self, input_dim, latent_dim, num_domains, n_species):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input data x.
        latent_dim : int
            Dimensionality of the latent space z.
        num_domains : int
            Number of domains (one decoder per domain).
        n_species : int
            Number of species (conditions the latent prior).
        """
        super().__init__(input_dim, latent_dim, num_domains)
        self.n_species = n_species
        self.prior = ConditionalPrior(n_species=n_species, latent_dim=latent_dim)

    def elbo_loss(self, x, mu, logvar, z, domain_id, species_id, beta=1.0):
        """
        Computes the ELBO loss against a species-conditional prior.

        Unlike the standard-normal-prior ELBO in the base class, the KL
        term here measures q(z | x) against p(z | species) rather than
        a fixed N(0, I) prior.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        mu : torch.Tensor
            Mean of q(z | x), as returned by `forward`.
        logvar : torch.Tensor
            Log-variance of q(z | x), as returned by `forward`.
        z : torch.Tensor
            Sampled latent representation, as returned by `forward`.
        domain_id : torch.Tensor
            Domain identifiers selecting the decoder used for the
            reconstruction term.
        species_id : torch.Tensor
            Per-sample species identifiers used to condition the prior
            p(z | species).
        beta : float, default=1.0
            Weight applied to the KL divergence term (beta-VAE annealing).

        Returns
        -------
        loss : torch.Tensor
            Mean negative ELBO (NLL) over the batch.
        recon_loss : torch.Tensor
            Mean reconstruction loss (negative log-likelihood) over the
            batch.
        kl_loss : torch.Tensor
            Mean KL divergence between q(z | x) and p(z | species) over
            the batch.
        """
        # Log-likelihood p(x|z,d)
        RE = self.decoder.log_prob(x, z, domain_id)

        # Build species one-hot
        species_onehot = F.one_hot(species_id, num_classes=self.n_species).float()

        # Prior p(z | species)
        mu_p, logvar_p = self.prior(species_onehot)

        # KL(q(z|x) || p(z|species))
        KL = -0.5 * torch.sum(1 + (logvar - logvar_p) - ((mu - mu_p) ** 2 + logvar.exp()) / logvar_p.exp(), dim=1)

        NLL = -(RE - beta * KL)
        return NLL.mean(), (-RE).mean(), KL.mean()

    def fgsm_perturb(self, x, domain_id, species_id, eps, beta=1.0):
        """
        Generates FGSM adversarial perturbation of x.

        x_adv = x + eps * sign(∇_x L_elbo(x))

        The perturbation is clipped to keep x_adv in the valid
        input range [0, 1] (row-minmax normalized spectra).

        Parameters
        ----------
        x : torch.Tensor
            Input data to perturb.
        domain_id : torch.Tensor
            Domain identifiers selecting the decoder used to compute the
            loss whose gradient drives the perturbation.
        species_id : torch.Tensor
            Per-sample species identifiers used to condition the prior
            p(z | species) when computing the loss.
        eps : float
            L-inf perturbation budget.
        beta : float, default=1.0
            Weight applied to the KL divergence term when computing the
            loss used to generate the perturbation.

        Returns
        -------
        torch.Tensor
            Adversarially perturbed input x_adv, detached from the graph
            and clamped to [0, 1].
        """
        x_adv = x.clone().detach().requires_grad_(True)

        mu, logvar, z = self.forward(x_adv, domain_id)
        loss, _, _ = self.elbo_loss(x_adv, mu, logvar, z, domain_id=domain_id, species_id=species_id, beta=beta)
        loss.backward()

        with torch.no_grad():
            x_adv = x + eps * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv.detach()

    def pgd_perturb(self, x, domain_id, species_id, eps, alpha, steps, beta=1.0):
        """
        Generates a PGD adversarial perturbation of x against the ELBO loss.

        Iteratively applies FGSM steps of size alpha, projecting back onto
        the L-inf ball of radius eps around the original input after each step.
        Initialized from a random point inside the ball (random_start=True).

        Parameters
        ----------
        x : torch.Tensor
            Input data to perturb.
        domain_id : torch.Tensor
            Domain identifiers selecting the decoder used to compute the
            loss whose gradient drives the perturbation.
        species_id : torch.Tensor
            Per-sample species identifiers used to condition the prior
            p(z | species) when computing the loss.
        eps : float
            L-inf perturbation budget (radius of the allowed ball).
        alpha : float
            Step size per iteration. Typically set to (eps / steps) * 2.
        steps : int
            Number of PGD iterations.
        beta : float, default=1.0
            Weight applied to the KL divergence term when computing the
            loss used to generate the perturbation.

        Returns
        -------
        torch.Tensor
            Adversarially perturbed input x_adv, detached from the graph
            and clamped to [0, 1].
        """
        x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
        for _ in range(steps):
            x_adv.requires_grad_(True)
            mu, logvar, z = self.forward(x_adv, domain_id)
            loss, _, _ = self.elbo_loss(x_adv, mu, logvar, z, domain_id=domain_id, species_id=species_id, beta=beta)
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - x, min=-eps, max=eps)
                x_adv = torch.clamp(x + delta, 0, 1)
        return x_adv.detach()

class MultiVAE_Bernoulli_SpeciesPrior_Adv_Extended(MultiVAE_Bernoulli_SpeciesPrior_Adv):
    def __init__(self, input_dim, latent_dim, num_domains, n_species, epochs=100, lr=1e-4, annealing_epochs=50, patience=20, adv_eps=0.02, adv_lambda=0.5, adv_attack="fgsm", pgd_steps=7, pgd_alpha=None):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input data x.
        latent_dim : int
            Dimensionality of the latent space z.
        num_domains : int
            Number of acquisition domains (one decoder per domain).
        n_species : int
            Number of species (conditions the latent prior).
        epochs : int, default=100
            Maximum number of training epochs.
        lr : float, default=1e-4
            Learning rate for the Adam optimizer.
        annealing_epochs : int, default=50
            Unused by trainloop (beta is fixed at 1.0); kept for interface
            compatibility with the base class.
        patience : int, default=20
            Number of epochs without validation-loss improvement before
            early stopping is triggered.
        adv_eps : float, default=0.02
            L-inf perturbation budget in row-minmax normalized space [0, 1].
        adv_lambda : float, default=0.5
            Mixing coefficient between clean and adversarial loss.
            0.0 = standard training, 1.0 = fully adversarial.
        adv_attack : str, default="fgsm"
            Attack used to generate perturbations during training.
            One of "fgsm" or "pgd".
        pgd_steps : int, default=7
            Number of PGD iterations per batch. Only used when adv_attack="pgd".
        pgd_alpha : float or None, default=None
            PGD step size per iteration. If None, set to (adv_eps / pgd_steps) * 2.
            Only used when adv_attack="pgd".
        """
        super().__init__(input_dim, latent_dim, num_domains, n_species)
        self.epochs = epochs
        self.lr = lr
        self.annealing_epochs = annealing_epochs
        self.patience = patience
        self.adv_eps = adv_eps
        self.adv_lambda = adv_lambda
        self.adv_attack = adv_attack  
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha if pgd_alpha is not None else (adv_eps / pgd_steps) * 2

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

        self.loss_during_training    = []
        self.reconstruc_during_training = []
        self.KL_during_training      = []
        
    def _perturb(self, x, domain_id, species_id, beta):
        """
        Generates an adversarial perturbation of x using the configured attack.

        Dispatches to fgsm_perturb or pgd_perturb depending on self.adv_attack.

        Parameters
        ----------
        x : torch.Tensor
            Input data to perturb.
        domain_id : torch.Tensor
            Domain identifiers selecting the decoder for the ELBO loss.
        species_id : torch.Tensor
            Per-sample species identifiers for the conditional prior.
        beta : float
            Weight applied to the KL divergence term in the ELBO loss.

        Returns
        -------
        torch.Tensor
            Adversarially perturbed input, detached and clamped to [0, 1].
        """
        if self.adv_attack == "pgd":
            return self.pgd_perturb(x, domain_id, species_id, eps=self.adv_eps, alpha=self.pgd_alpha, steps=self.pgd_steps, beta=beta)
        return self.fgsm_perturb(x, domain_id, species_id, eps=self.adv_eps, beta=beta)

    def trainloop(self, trainloader, validloader, device):
        """
        Trains the model with adversarial (FGSM) data augmentation and early stopping.

        Runs up to `self.epochs` epochs. For each training batch, a clean
        loss and an FGSM-adversarial loss (via `fgsm_perturb`) are
        computed and mixed as
        `(1 - adv_lambda) * L_clean + adv_lambda * L_adv`. Evaluates on
        `validloader` after each epoch using only clean data, and reverts
        to the best-validation-loss weights found if early stopping
        triggers before the last epoch. Per-epoch (train, val) tuples for
        total loss, reconstruction loss, and KL are appended to
        `self.loss_during_training`, `self.reconstruc_during_training`,
        and `self.KL_during_training` respectively.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            Yields (x, domain_id, species_id) batches.
        validloader : torch.utils.data.DataLoader
            Same batch format as `trainloader`, used for validation and
            early stopping.
        device : torch.device or str
            Device to move the model and batches to.

        Returns
        -------
        None
            The model is updated in place; on early stopping it is
            reloaded with the best validation-loss weights found.
        """
        self.to(device)

        best_val_loss    = float("inf")
        patience_counter = 0
        best_state       = None

        print(
            f"Adversarial training | eps={self.adv_eps} | lambda={self.adv_lambda} | "
            f"epochs={self.epochs} | patience={self.patience}"
        )

        for epoch in range(self.epochs):
            beta = 1.0
            self.train()
            tr_loss, tr_recon, tr_kl = 0.0, 0.0, 0.0

            for batch in trainloader:
                x, domain_id, species_id = batch[0], batch[1], batch[2]
                x         = x.to(device)
                domain_id = domain_id.to(device)
                species_id = species_id.to(device)

                self.optimizer.zero_grad()
                mu, logvar, z      = self.forward(x, domain_id)
                loss_clean, recon, kl = self.elbo_loss(x, mu, logvar, z, domain_id=domain_id, species_id=species_id, beta=beta)

                x_adv = self._perturb(x, domain_id, species_id, beta=beta)

                mu_adv, logvar_adv, z_adv = self.forward(x_adv, domain_id)
                loss_adv, _, _ = self.elbo_loss(x_adv, mu_adv, logvar_adv, z_adv, domain_id=domain_id, species_id=species_id, beta=beta)

                loss = (1.0 - self.adv_lambda) * loss_clean + self.adv_lambda * loss_adv
                loss.backward()
                self.optimizer.step()

                tr_loss  += loss.item()
                tr_recon += recon.item()
                tr_kl    += kl.item()

            tr_loss  /= len(trainloader)
            tr_recon /= len(trainloader)
            tr_kl    /= len(trainloader)

            self.eval()
            val_loss, val_recon, val_kl = 0.0, 0.0, 0.0

            with torch.no_grad():
                for batch in validloader:
                    x, domain_id, species_id = batch[0], batch[1], batch[2]
                    x          = x.to(device)
                    domain_id  = domain_id.to(device)
                    species_id = species_id.to(device)

                    mu, logvar, z = self.forward(x, domain_id)
                    loss, recon, kl = self.elbo_loss(x, mu, logvar, z, domain_id=domain_id, species_id=species_id, beta=beta)
                    val_loss  += loss.item()
                    val_recon += recon.item()
                    val_kl    += kl.item()

            val_loss  /= len(validloader)
            val_recon /= len(validloader)
            val_kl    /= len(validloader)

            self.loss_during_training.append((tr_loss, val_loss))
            self.reconstruc_during_training.append((tr_recon, val_recon))
            self.KL_during_training.append((tr_kl, val_kl))

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | beta={beta:.3f} | "
                    f"[Train] Loss={tr_loss:.4f} | Recon={tr_recon:.4f} | KL={tr_kl:.4f} || "
                    f"[Val] Loss={val_loss:.4f} | Recon={val_recon:.4f} | KL={val_kl:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_state       = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.load_state_dict(best_state)
