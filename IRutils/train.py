import os

import torch
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from IRutils.validate import validate, validate_amp


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
            query_inputs = batch["query_input_ids"].to(device)
            query_masks = batch["query_attention_mask"].to(device)
            positive_inputs = batch["pos_doc_input_ids"].to(device)
            positive_masks = batch["pos_doc_attention_mask"].to(device)
            negative_inputs = batch["neg_doc_input_ids"].to(device)
            negative_masks = batch["neg_doc_attention_mask"].to(device)

            optimizer.zero_grad()

            try:
                query_embeddings = model.get_embedding(query_inputs, query_masks)
                positive_embeddings = model.get_embedding(positive_inputs, positive_masks)
                negative_embeddings = model.get_embedding(negative_inputs, negative_masks)
            except AttributeError:
                print(
                    "Warning: model.get_embedding() not found, falling back to model(). Ensure model() returns corpus_embeddings.")
                query_embeddings = model(query_inputs, query_masks)
                positive_embeddings = model(positive_inputs, positive_masks)
                negative_embeddings = model(negative_inputs, negative_masks)

            # Calculate triplet loss
            positive_distances = torch.norm(query_embeddings - positive_embeddings, p=2, dim=1)
            negative_distances = torch.norm(query_embeddings - negative_embeddings, p=2, dim=1)

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


def train_triplet_ranker_amp(model, train_loader, val_loader, optimizer, device, model_path,
                         epochs=10, patience=3, margin=1.0): # Add margin parameter
    """
    Trains the TripletRankerModel with early stopping, optional AMP,
    and the specific triplet loss calculation.
    """
    print(f"Training on device: {device}")
    best_val_loss = float('inf')
    patience_counter = 0
    loss_margin = margin # Use the passed margin

    # --- AMP Setup ---
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    print(f"Using Automatic Mixed Precision (AMP): {use_amp}")
    # --- End AMP Setup ---

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        # --- Training Loop ---
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)"):
            # Move batch data to device
            query_inputs = batch["query_input_ids"].to(device)
            query_masks = batch["query_attention_mask"].to(device)
            positive_inputs = batch["pos_doc_input_ids"].to(device)
            positive_masks = batch["pos_doc_attention_mask"].to(device)
            negative_inputs = batch["neg_doc_input_ids"].to(device)
            negative_masks = batch["neg_doc_attention_mask"].to(device)

            optimizer.zero_grad()

            # --- AMP: Forward pass with autocast ---
            with autocast('cuda', enabled=use_amp):
                # Ensure you use the same method (model() or get_embedding()) as in validate
                # Using get_embedding here for consistency with validate's preference
                try:
                    query_embeddings = model.get_embedding(query_inputs, query_masks)
                    positive_embeddings = model.get_embedding(positive_inputs, positive_masks)
                    negative_embeddings = model.get_embedding(negative_inputs, negative_masks)
                except AttributeError:
                     print("Warning: model.get_embedding() not found, falling back to model(). Ensure model() returns corpus_embeddings.")
                     query_embeddings = model(query_inputs, query_masks)
                     positive_embeddings = model(positive_inputs, positive_masks)
                     negative_embeddings = model(negative_inputs, negative_masks)

                # --- Calculate triplet loss EXACTLY as in validate.py ---
                positive_distances = torch.norm(query_embeddings - positive_embeddings, p=2, dim=1)
                negative_distances = torch.norm(query_embeddings - negative_embeddings, p=2, dim=1)
                loss = torch.relu(positive_distances - negative_distances + loss_margin).mean()
                # --- End loss calculation ---
            # --- End AMP autocast block ---

            # Scale loss, backward, step optimizer, update scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        print(f"\nEpoch {epoch+1}/{epochs}, Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        # Call the updated validate function, passing use_amp
        avg_val_loss = validate_amp(model, val_loader, device, margin=loss_margin, use_amp=use_amp)

        # Early Stopping and Model Saving Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss improved. Saving model to {model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete!")
    # Load the best model state before returning
    print(f"Loading best model from {model_path} based on validation loss ({best_val_loss:.4f}).")
    # Ensure the file exists before trying to load
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Best model file not found at {model_path}. Returning current model state.")
    model.eval() # Set to eval mode before returning
    return model