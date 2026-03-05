from torch.utils.data import TensorDataset, DataLoader


def construct_dataloaders(X_train_tensor, X_val_tensor, X_all_tensor, domain_train_tensor, domain_val_tensor, domain_all_tensor, batch_size, species_train_tensor=None, species_val_tensor=None, species_all_tensor=None, amr_train_tensor=None, amr_val_tensor=None, amr_all_tensor=None):
    """
    Construct PyTorch DataLoaders for training, validation and full datasets.

    Optionally includes species labels for multi-task or conditional models.

    Parameters
    ----------
    X_train_tensor : torch.Tensor
        Training feature tensor (float32).
    X_val_tensor : torch.Tensor
        Validation feature tensor (float32).
    X_all_tensor : torch.Tensor
        Feature tensor containing all samples.
    domain_train_tensor : torch.Tensor
        Domain IDs for training samples (long).
    domain_val_tensor : torch.Tensor
        Domain IDs for validation samples (long).
    domain_all_tensor : torch.Tensor
        Domain IDs for all samples (long).
    batch_size : int
        Batch size for training and validation loaders.
    species_train_tensor : torch.Tensor or None, optional
        Species labels for training samples.
    species_val_tensor : torch.Tensor or None, optional
        Species labels for validation samples.
    species_all_tensor : torch.Tensor or None, optional
        Species labels for all samples.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data.
    all_loader : torch.utils.data.DataLoader
        DataLoader for the full dataset.
    """ 

    if species_train_tensor is None and amr_train_tensor is None:
        train_dataset = TensorDataset(X_train_tensor, domain_train_tensor)
        val_dataset   = TensorDataset(X_val_tensor, domain_val_tensor)
        all_dataset   = TensorDataset(X_all_tensor, domain_all_tensor)
    elif amr_train_tensor is None:
        train_dataset = TensorDataset(X_train_tensor, domain_train_tensor, species_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, domain_val_tensor, species_val_tensor)
        all_dataset = TensorDataset(X_all_tensor, domain_all_tensor, species_all_tensor)
    else:
        train_dataset = TensorDataset(X_train_tensor, domain_train_tensor, species_train_tensor, amr_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, domain_val_tensor, species_val_tensor, amr_val_tensor)
        all_dataset = TensorDataset(X_all_tensor, domain_all_tensor, species_all_tensor, amr_all_tensor)
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    all_loader  = DataLoader(
        all_dataset, 
        batch_size=256, 
        shuffle=False)

    return train_loader, val_loader, all_loader
