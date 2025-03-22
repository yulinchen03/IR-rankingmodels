import torch
from tqdm.notebook import tqdm

from IRutils.validate import validate


def train_triplet_ranker(model, train_loader, val_loader, optimizer, device, save_path, epochs=10, margin=1.0, patience=3):
    """
    Trains a TripletRankerModel with validation and early stopping.

    Args:
        model: The TripletRankerModel to train.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        optimizer: Optimizer for training.
        device: Device to train on (e.g., 'cuda' or 'cpu').
        epochs: Number of training epochs.
        margin: Margin for the triplet loss.
        patience: Number of epochs to wait for improvement before early stopping.
    """
    model.train()
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    epoch = 0

    while epoch < epochs:
        total_loss = 0.
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Training)"):
            anchor_inputs = batch["anchor_input_ids"].to(device)
            anchor_masks = batch["anchor_attention_mask"].to(device)
            positive_inputs = batch["positive_input_ids"].to(device)
            positive_masks = batch["positive_attention_mask"].to(device)
            negative_inputs = batch["negative_input_ids"].to(device)
            negative_masks = batch["negative_attention_mask"].to(device)

            optimizer.zero_grad()

            anchor_embeddings = model(anchor_inputs, anchor_masks)
            positive_embeddings = model(positive_inputs, positive_masks)
            negative_embeddings = model(negative_inputs, negative_masks)

            # Calculate triplet loss
            positive_distances = torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=1)
            negative_distances = torch.norm(anchor_embeddings - negative_embeddings, p=2, dim=1)

            loss = torch.relu(positive_distances - negative_distances + margin).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_loss:.4f}")

        # Validation Phase
        val_loss = validate(model, val_loader, device, margin)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)  # save best model.
            print("Validation loss improved. Saving model.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        epoch += 1

    # Load best model
    model.load_state_dict(torch.load(save_path))
    print("Loaded best model based on validation loss.")

    print("Training complete!")
    return model