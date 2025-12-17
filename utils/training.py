def train_model(model, train_loader, val_loader, device):
    model.trainloop(train_loader, val_loader, device=device)
    return model
