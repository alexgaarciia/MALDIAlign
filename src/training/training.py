import inspect


def train_model(model, train_loader, val_loader, device, species_weights=None, pos_weight=None):
    """
    Universal training wrapper that supports both weighted and unweighted models.
    Automatically checks whether the model's trainloop supports `species_weights`.
    """

    # Store pos_weight in the model so the loss can access it
    model.pos_weight = pos_weight

    sig = inspect.signature(model.trainloop)

    if "species_weights" in sig.parameters:
        model.trainloop(
            train_loader,
            val_loader,
            device=device,
            species_weights=species_weights
        )
    else:
        model.trainloop(
            train_loader,
            val_loader,
            device=device
        )

    return model
