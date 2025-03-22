import torch
from ir_measures import calc_aggregate
from tqdm import tqdm
from ir_measures import nDCG, AP, P, R, RR


def evaluate(model, test_loader, device, qrels):
    model.eval()
    model.to(device)
    run = {}  # Format: {qid: {doc_id: score}}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            qids = batch["qid"]
            pos_dids = batch["pos_did"]
            neg_dids = batch["neg_did"]

            # Process embeddings and calculate distances
            anchor_inputs = batch["anchor_input_ids"].to(device)
            anchor_masks = batch["anchor_attention_mask"].to(device)
            positive_inputs = batch["positive_input_ids"].to(device)
            positive_masks = batch["positive_attention_mask"].to(device)
            negative_inputs = batch["negative_input_ids"].to(device)
            negative_masks = batch["negative_attention_mask"].to(device)

            anchor_embeddings = model(anchor_inputs, anchor_masks)
            positive_embeddings = model(positive_inputs, positive_masks)
            negative_embeddings = model(negative_inputs, negative_masks)

            pos_distances = torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=1)
            neg_distances = torch.norm(anchor_embeddings - negative_embeddings, p=2, dim=1)

            # Build the run dictionary
            i = 0
            while i < len(qids):
                qid = qids[i]
                pos_did = pos_dids[i]
                neg_did = neg_dids[i]

                pos_score = -pos_distances[i].item()
                neg_score = -neg_distances[i].item()

                if qid not in run:
                    run[qid] = {}

                # Add scores directly (no list of dicts)
                run[qid][pos_did] = pos_score
                run[qid][neg_did] = neg_score

                i += 1

    # Calculate metrics
    metrics = [
        nDCG @ 10, nDCG @ 100,
        AP @ 10, AP @ 100,
        P @ 10, R @ 10,
        P @ 100, R @ 100,
        RR
    ]

    metric_scores = calc_aggregate(metrics, qrels, run)

    return metric_scores