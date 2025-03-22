import torch
from tqdm import tqdm


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
            anchor_inputs = batch["anchor_input_ids"].to(device)
            anchor_masks = batch["anchor_attention_mask"].to(device)
            positive_inputs = batch["positive_input_ids"].to(device)
            positive_masks = batch["positive_attention_mask"].to(device)
            negative_inputs = batch["negative_input_ids"].to(device)
            negative_masks = batch["negative_attention_mask"].to(device)

            anchor_embeddings = model(anchor_inputs, anchor_masks)
            positive_embeddings = model(positive_inputs, positive_masks)
            negative_embeddings = model(negative_inputs, negative_masks)

            positive_distances = torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=1)
            negative_distances = torch.norm(anchor_embeddings - negative_embeddings, p=2, dim=1)

            loss = torch.relu(positive_distances - negative_distances + margin).mean()
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss