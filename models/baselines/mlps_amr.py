import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


############################################################
# LINEAR PROBE
############################################################
class LinearProbe_Extended(nn.Module):
    """
    A simple linear probe for evaluating latent representations.
    Trains a single fully-connected layer on top of frozen embeddings
    using cross-entropy loss with early stopping.
    """

    def __init__(self, latent_dim, n_species, epochs=50, lr=1e-3, patience=10):
        """
        Parameters
        ----------
        latent_dim : int
            Dimensionality of the input embeddings z.
        n_species : int
            Number of species classes (output dimension of the probe).
        epochs : int, default=50
            Maximum number of training epochs.
        lr : float, default=1e-3
            Learning rate for the Adam optimizer.
        patience : int, default=10
            Number of epochs without validation-loss improvement before
            early stopping is triggered.
        """
        super(LinearProbe_Extended, self).__init__()
        self.latent_dim = latent_dim
        self.n_species = n_species
        self.epochs = epochs
        self.lr = lr
        self.patience = patience

        self.fc = nn.Linear(latent_dim, n_species)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss_during_training = []

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input embeddings, shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            Species classification logits, shape (batch_size, n_species).
        """
        return self.fc(x)

    def trainloop(self, trainloader, validloader, device):
        """
        Trains the probe with cross-entropy loss and early stopping.

        Runs up to `self.epochs` epochs, evaluating on `validloader`
        after each one, and reverts to the best-validation-loss weights
        found if early stopping triggers before the last epoch.
        Per-epoch (train, val) loss tuples are appended to
        `self.loss_during_training`.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            Yields (X_batch, y_batch) batches, with y_batch as species
            class indices.
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
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0
            for X_batch, y_batch in trainloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                self.optimizer.zero_grad()
                logits = self(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(trainloader)

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in validloader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = self(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()
            val_loss /= len(validloader)

            self.loss_during_training.append((train_loss, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.load_state_dict(best_state)


############################################################
# AMR PROBES
############################################################
class AMRProbeRaw(nn.Module):
    """
    Raw spectra AMR classifier with species-conditioned prediction heads.

    The spectrum is first encoded through a shared trunk (same architecture as
    the VAE encoder), then the species embedding is concatenated to the trunk
    output before the antibiotic prediction heads:

        h_i = trunk(x_i)
        u_s = species_emb(s_i)
        ŷ_ij = sigmoid(head_j([h_i, u_s]))

    This is in contrast to concatenating the species embedding to the raw
    spectrum before the trunk, which would force the trunk to process
    species information at every layer.

    Trained end-to-end with masked BCEWithLogitsLoss (NaN labels ignored).
    """

    def __init__(self, input_dim, n_species, n_antibiotics, species_emb_dim=128, epochs=50, lr=1e-3, patience=10):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input spectrum x.
        n_species : int
            Number of species (used to size the species embedding).
        n_antibiotics : int
            Number of antibiotics (one prediction head per antibiotic).
        species_emb_dim : int, default=128
            Dimensionality of the learned species embedding, concatenated
            to the trunk output before each head.
        epochs : int, default=50
            Maximum number of training epochs.
        lr : float, default=1e-3
            Learning rate for the Adam optimizer.
        patience : int, default=10
            Number of epochs without validation-loss improvement before
            early stopping is triggered.
        """
        super().__init__()
        self.n_antibiotics = n_antibiotics
        self.n_species = n_species
        self.species_emb_dim = species_emb_dim
        self.epochs = epochs
        self.lr = lr
        self.patience = patience

        self.trunk = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())

        self.species_emb = nn.Embedding(n_species, species_emb_dim)
        self.heads = nn.ModuleList([nn.Linear(128 + species_emb_dim, 1) for _ in range(n_antibiotics)])

    def forward(self, x, species_id):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input spectrum, shape (batch_size, input_dim).
        species_id : torch.Tensor
            Per-sample species identifiers, embedded via
            `self.species_emb`.

        Returns
        -------
        torch.Tensor
            AMR prediction logits, shape (batch_size, n_antibiotics).
        """
        h = self.trunk(x)
        u_s = self.species_emb(species_id)
        z = torch.cat([h, u_s], dim=1)
        return torch.cat([head(z) for head in self.heads], dim=1)

    def trainloop(self, X_tr, y_tr, X_va, y_va, species_tr, species_va, device):
        """
        Trains the model with full-batch gradient descent and early stopping.

        Runs up to `self.epochs` epochs, each performing a single
        full-batch gradient step over `(X_tr, y_tr, species_tr)` and
        evaluating on `(X_va, y_va, species_va)`. Labels may contain
        NaN, which are masked out of the BCE loss. Reverts to the
        best-validation-loss weights found if early stopping triggers
        before the last epoch.

        Parameters
        ----------
        X_tr : array-like
            Training spectra, shape (n_train, input_dim).
        y_tr : array-like
            Training AMR labels, shape (n_train, n_antibiotics). May
            contain NaN for missing labels.
        X_va : array-like
            Validation spectra, shape (n_val, input_dim).
        y_va : array-like
            Validation AMR labels, shape (n_val, n_antibiotics). May
            contain NaN for missing labels.
        species_tr : array-like
            Training species identifiers, shape (n_train,).
        species_va : array-like
            Validation species identifiers, shape (n_val,).
        device : torch.device or str
            Device to move the model and data to.

        Returns
        -------
        None
            The model is updated in place; on early stopping it is
            reloaded with the best validation-loss weights found.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        sp_tr_t = torch.tensor(species_tr, dtype=torch.long)
        X_va_t = torch.tensor(X_va, dtype=torch.float32)
        y_va_t = torch.tensor(y_va, dtype=torch.float32)
        sp_va_t = torch.tensor(species_va, dtype=torch.long)

        best_val, best_state, patience_counter = float("inf"), None, 0
        self.to(device)

        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()
            logits = self(X_tr_t.to(device), sp_tr_t.to(device))
            y_b = y_tr_t.to(device)
            mask = ~torch.isnan(y_b)
            loss = (criterion(logits, torch.nan_to_num(y_b, 0.0)) * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_logits = self(X_va_t.to(device), sp_va_t.to(device))
                y_vb = y_va_t.to(device)
                mask_v = ~torch.isnan(y_vb)
                val_loss = (criterion(val_logits, torch.nan_to_num(y_vb, 0.0)) * mask_v).sum() / mask_v.sum().clamp(min=1)

            if val_loss.item() < best_val:
                best_val = val_loss.item()
                best_state = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.load_state_dict(best_state)

    def predict_proba(self, X, species, device, batch_size=512):
        """
        Predicts AMR probabilities in batches.

        Parameters
        ----------
        X : array-like
            Input spectra, shape (n_samples, input_dim).
        species : array-like
            Species identifiers, shape (n_samples,).
        device : torch.device or str
            Device to run inference on.
        batch_size : int, default=512
            Number of samples per inference batch.

        Returns
        -------
        numpy.ndarray
            Predicted AMR probabilities, shape (n_samples, n_antibiotics).
        """
        self.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        sp_t = torch.tensor(species, dtype=torch.long)
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                x_b = X_t[i:i+batch_size].to(device)
                sp_b = sp_t[i:i+batch_size].to(device)
                logits = self(x_b, sp_b)
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(probs_list)


class AMRProbeLatent(nn.Module):
    """
    Linear AMR probe on VAE latent representations.

    Takes the frozen latent z (latent_dim) and applies independent linear
    heads for each antibiotic. No hidden layers — this is a pure linear
    probe that measures whether AMR information is linearly accessible
    in the VAE's latent space without any additional capacity.

    Trained with masked BCEWithLogitsLoss.
    """

    def __init__(self, latent_dim, n_antibiotics, epochs=50, lr=1e-3, patience=10):
        """
        Parameters
        ----------
        latent_dim : int
            Dimensionality of the input latent representation z.
        n_antibiotics : int
            Number of antibiotics (one linear head per antibiotic).
        epochs : int, default=50
            Maximum number of training epochs.
        lr : float, default=1e-3
            Learning rate for the Adam optimizer.
        patience : int, default=10
            Number of epochs without validation-loss improvement before
            early stopping is triggered.
        """
        super().__init__()
        self.n_antibiotics = n_antibiotics
        self.epochs = epochs
        self.lr = lr
        self.patience = patience

        self.heads = nn.ModuleList([nn.Linear(latent_dim, 1) for _ in range(n_antibiotics)])

    def forward(self, z):
        """
        Parameters
        ----------
        z : torch.Tensor
            Latent representation, shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            AMR prediction logits, shape (batch_size, n_antibiotics).
        """
        return torch.cat([head(z) for head in self.heads], dim=1)

    def trainloop(self, X_tr, y_tr, X_va, y_va, device):
        """
        Trains the probe with full-batch gradient descent and early stopping.

        Runs up to `self.epochs` epochs, each performing a single
        full-batch gradient step over `(X_tr, y_tr)` and evaluating on
        `(X_va, y_va)`. Labels may contain NaN, which are masked out of
        the BCE loss. Reverts to the best-validation-loss weights found
        if early stopping triggers before the last epoch.

        Parameters
        ----------
        X_tr : array-like
            Training latent representations, shape (n_train, latent_dim).
        y_tr : array-like
            Training AMR labels, shape (n_train, n_antibiotics). May
            contain NaN for missing labels.
        X_va : array-like
            Validation latent representations, shape (n_val, latent_dim).
        y_va : array-like
            Validation AMR labels, shape (n_val, n_antibiotics). May
            contain NaN for missing labels.
        device : torch.device or str
            Device to move the model and data to.

        Returns
        -------
        None
            The model is updated in place; on early stopping it is
            reloaded with the best validation-loss weights found.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        X_va_t = torch.tensor(X_va, dtype=torch.float32)
        y_va_t = torch.tensor(y_va, dtype=torch.float32)

        best_val = float("inf")
        best_state = None
        patience_counter = 0

        self.to(device)

        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()
            logits = self(X_tr_t.to(device))
            y_b = y_tr_t.to(device)
            mask = ~torch.isnan(y_b)
            loss = (criterion(logits, torch.nan_to_num(y_b, 0.0)) * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_logits = self(X_va_t.to(device))
                y_vb = y_va_t.to(device)
                mask_v = ~torch.isnan(y_vb)
                val_loss = (criterion(val_logits, torch.nan_to_num(y_vb, 0.0)) * mask_v).sum() / mask_v.sum().clamp(min=1)

            if val_loss.item() < best_val:
                best_val = val_loss.item()
                best_state = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.load_state_dict(best_state)

    def predict_proba(self, X, device, batch_size=512):
        """
        Predicts AMR probabilities in batches.

        Parameters
        ----------
        X : array-like
            Input latent representations, shape (n_samples, latent_dim).
        device : torch.device or str
            Device to run inference on.
        batch_size : int, default=512
            Number of samples per inference batch.

        Returns
        -------
        numpy.ndarray
            Predicted AMR probabilities, shape (n_samples, n_antibiotics).
        """
        self.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                logits = self(X_t[i:i+batch_size].to(device))
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(probs_list)


class AMRProbeRawNoSpecies(nn.Module):
    """
    MLP trained per species on raw spectra, without species onehot.
    Same trunk architecture as AMRProbeRaw but input is only the spectrum.
    """

    def __init__(self, n_antibiotics, epochs=50, lr=1e-3, patience=10):
        """
        Parameters
        ----------
        n_antibiotics : int
            Number of antibiotics (one prediction head per antibiotic).
        epochs : int, default=50
            Maximum number of training epochs.
        lr : float, default=1e-3
            Learning rate for the Adam optimizer.
        patience : int, default=10
            Number of epochs without validation-loss improvement before
            early stopping is triggered.

        Notes
        -----
        The trunk's input dimensionality is fixed at 6000, not
        configurable via this constructor.
        """
        super().__init__()
        self.n_antibiotics = n_antibiotics
        self.epochs = epochs
        self.lr = lr
        self.patience = patience

        self.trunk = nn.Sequential(nn.Linear(6000, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(n_antibiotics)])

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input spectrum, shape (batch_size, 6000).

        Returns
        -------
        torch.Tensor
            AMR prediction logits, shape (batch_size, n_antibiotics).
        """
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.heads], dim=1)

    def trainloop(self, X_tr, y_tr, X_va, y_va, device):
        """
        Trains the model with full-batch gradient descent and early stopping.

        Runs up to `self.epochs` epochs, each performing a single
        full-batch gradient step over `(X_tr, y_tr)` and evaluating on
        `(X_va, y_va)`. Labels may contain NaN, which are masked out of
        the BCE loss. Reverts to the best-validation-loss weights found
        if early stopping triggers before the last epoch.

        Parameters
        ----------
        X_tr : array-like
            Training spectra, shape (n_train, 6000).
        y_tr : array-like
            Training AMR labels, shape (n_train, n_antibiotics). May
            contain NaN for missing labels.
        X_va : array-like
            Validation spectra, shape (n_val, 6000).
        y_va : array-like
            Validation AMR labels, shape (n_val, n_antibiotics). May
            contain NaN for missing labels.
        device : torch.device or str
            Device to move the model and data to.

        Returns
        -------
        None
            The model is updated in place; on early stopping it is
            reloaded with the best validation-loss weights found.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        X_va_t = torch.tensor(X_va, dtype=torch.float32)
        y_va_t = torch.tensor(y_va, dtype=torch.float32)
        best_val, best_state, patience_counter = float("inf"), None, 0
        self.to(device)
        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()
            logits = self(X_tr_t.to(device))
            y_b = y_tr_t.to(device)
            mask = ~torch.isnan(y_b)
            loss = (criterion(logits, torch.nan_to_num(y_b, 0.0)) * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            optimizer.step()
            self.eval()
            with torch.no_grad():
                val_logits = self(X_va_t.to(device))
                y_vb = y_va_t.to(device)
                mask_v = ~torch.isnan(y_vb)
                val_loss = (criterion(val_logits, torch.nan_to_num(y_vb, 0.0)) * mask_v).sum() / mask_v.sum().clamp(min=1)
            if val_loss.item() < best_val:
                best_val = val_loss.item()
                best_state = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                break
        if best_state is not None:
            self.load_state_dict(best_state)

    def predict_proba(self, X, device, batch_size=512):
        """
        Predicts AMR probabilities in batches.

        Parameters
        ----------
        X : array-like
            Input spectra, shape (n_samples, 6000).
        device : torch.device or str
            Device to run inference on.
        batch_size : int, default=512
            Number of samples per inference batch.

        Returns
        -------
        numpy.ndarray
            Predicted AMR probabilities, shape (n_samples, n_antibiotics).
        """
        self.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                logits = self(X_t[i:i+batch_size].to(device))
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(probs_list)
