import torch
from tqdm import tqdm
from torch.amp import autocast # <-- Import autocast

def validate(model, val_loader, device, margin=1.0):
    """
    Validates a TripletRankerModel.

    Args:
        model: The TripletRankerModel to validate.
        val_loader: DataLoader for the validation data.
        device: Device to validate on (e.g., 'cuda' or 'cpu').
        margin: Margin for the triplet loss.
    """

    model.eval()
    model.to(device)
    total_val_loss = 0.
    with torch.no_grad():

        for batch in tqdm(val_loader, desc="Validation"):
            query_inputs = batch["query_input_ids"].to(device)
            query_masks = batch["query_attention_mask"].to(device)
            positive_inputs = batch["pos_doc_input_ids"].to(device)
            positive_masks = batch["pos_doc_attention_mask"].to(device)
            negative_inputs = batch["neg_doc_input_ids"].to(device)
            negative_masks = batch["neg_doc_attention_mask"].to(device)

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
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss


def validate_amp(model, val_loader, device, margin=1.0, use_amp=False): # <-- Add use_amp flag
    """
    Validates a TripletRankerModel with optional AMP.
    Assumes model() or model.get_embedding() returns corpus_embeddings.

    Args:
        model: The TripletRankerModel to validate.
        val_loader: DataLoader for the validation data.
        device: Device to validate on (e.g., 'cuda' or 'cpu').
        margin: Margin for the triplet loss.
        use_amp: Boolean flag to enable/disable AMP autocast. # <-- Add description
    """
    model.eval()
    model.to(device)
    total_val_loss = 0.
    num_val_batches = 0
    with torch.no_grad():

        for batch in tqdm(val_loader, desc="Validation"):
            # Move batch data to device
            query_inputs = batch["query_input_ids"].to(device)
            query_masks = batch["query_attention_mask"].to(device)
            positive_inputs = batch["pos_doc_input_ids"].to(device)
            positive_masks = batch["pos_doc_attention_mask"].to(device)
            negative_inputs = batch["neg_doc_input_ids"].to(device)
            negative_masks = batch["neg_doc_attention_mask"].to(device)

            # --- Wrap forward pass and loss calculation with autocast ---
            with autocast('cuda', enabled=use_amp):
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

                positive_distances = torch.norm(query_embeddings - positive_embeddings, p=2, dim=1)
                negative_distances = torch.norm(query_embeddings - negative_embeddings, p=2, dim=1)

                # Calculate loss inside autocast context
                loss = torch.relu(positive_distances - negative_distances + margin).mean()
            # --- End autocast block ---

            total_val_loss += loss.item()
            num_val_batches += 1

    # Avoid division by zero if val_loader is empty
    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    print(f"\nValidation Loss: {avg_val_loss:.4f}")
    return avg_val_loss