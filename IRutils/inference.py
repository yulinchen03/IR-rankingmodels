import torch
from ir_measures import calc_aggregate
from tqdm import tqdm
from ir_measures import nDCG, AP, P, R, RR

def write_results(metric_scores, save_path, model_name, dataset_name, length_setting):
    # Save results to a file
    with open(save_path, "w") as f:
        f.write(f"Evaluation Results for {model_name} model finetuned on {length_setting} queries from {dataset_name} dataset:\n")
        f.write(f"normalized Discounted Cumulative Gain@10: {metric_scores[nDCG @ 10]:.4f}\n")
        f.write(f"normalized Discounted Cumulative Gain@100: {metric_scores[nDCG @ 100]:.4f}\n")
        f.write(f"\n")
        f.write(f"[Mean] Average Precision@10: {metric_scores[AP @ 10]:.4f}\n")
        f.write(f"[Mean] Average Precision@100: {metric_scores[AP @ 100]:.4f}\n")
        f.write(f"\n")
        f.write(f"Precision@10: {metric_scores[P @ 10]:.4f}\n")
        f.write(f"Recall@10: {metric_scores[R @ 10]:.4f}\n")
        f.write(f"\n")
        f.write(f"Precision@100: {metric_scores[P @ 100]:.4f}\n")
        f.write(f"Recall@100: {metric_scores[R @ 100]:.4f}\n")
        f.write(f"\n")
        f.write(f"[Mean] Reciprocal Rank: {metric_scores[RR]:.4f}\n")
        f.write(f"\n")
        f.write(f"----------------------------------------------------\n")
        f.write(f"\n")
        f.write(f"Explanation of metrics:\n")
        f.write(
            f"NDCG@k (Normalized Discounted Cumulative Gain: Ranking Quality | Prioritizes highly relevant documents appearing earlier in the ranking.\n")
        f.write(
            f"MAP (Mean Average Precision): Overall Relevance | Measures ranking precision across all relevant documents. Best for small-scale retrieval tasks.\n")
        f.write(
            f"Precision@k: Relevance | Measures how many of the top-k documents are relevant. Works well in precision-sensitive applications.\n")
        f.write(
            f"Recall@k: Coverage | Measures how many relevant documents appear in the top-k results. Important in recall-sensitive tasks.\n")
        f.write(
            f"MRR (Mean Reciprocal Rank): Single Relevant Result | Focuses on ranking the first relevant document. Good for QA tasks.\n")

    print(f'Successfully written results to {save_path}.')


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

def evaluate_average_ensemble(models, test_loader, device, qrels):
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

            pd = []
            nd = []

            for model in models.values():
                model.eval()
                model.to(device)

                anchor_embeddings = model(anchor_inputs, anchor_masks)
                positive_embeddings = model(positive_inputs, positive_masks)
                negative_embeddings = model(negative_inputs, negative_masks)

                pos_distances = torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=1)
                neg_distances = torch.norm(anchor_embeddings - negative_embeddings, p=2, dim=1)

                pd.append(pos_distances)
                nd.append(neg_distances)

            # Compute the average distances across all three models
            final_pos_distances = torch.stack(pd).mean(dim=0)  # Average over models
            final_neg_distances = torch.stack(nd).mean(dim=0)  # Average over models

            # Build the run dictionary
            i = 0
            while i < len(qids):
                qid = qids[i]
                pos_did = pos_dids[i]
                neg_did = neg_dids[i]

                pos_score = -final_pos_distances[i].item()
                neg_score = -final_neg_distances[i].item()

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


def evaluate_conditional_ensemble(models, t1, t2, test_loader, device, qrels, queries_test):
    """
    Evaluates using a conditional ensemble based on query length.

    Args:
        models: Models trained on varying query lengths.
        t1: Upper bound length for short queries (exclusive).
        t2: Upper bound length for medium queries (exclusive).
        test_loader: DataLoader for the test set.
        device: The device to run inference on (e.g., 'cuda').
        qrels: Ground truth relevance judgments.
        queries_test: Dictionary mapping qid to query text for the test set.

    Returns:
        Dictionary of metric scores.
    """
    for model in models:
        if 'short' in model:
            short_model = models[model]
        elif 'medium' in model:
            medium_model = models[model]
        elif 'long' in model:
            long_model = models[model]
        else:
            raise Exception("Unknown model detected!")

    short_model.eval().to(device)
    medium_model.eval().to(device)
    long_model.eval().to(device)

    run = {}  # Format: {qid: {doc_id: score}}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Conditional Ensemble"):
            qids = batch["qid"]
            pos_dids = batch["pos_did"]
            neg_dids = batch["neg_did"]

            # We need to process each item individually because model selection depends on query length
            for i in range(len(qids)):
                qid = qids[i]
                pos_did = pos_dids[i]
                neg_did = neg_dids[i]

                # Get query text and length
                query_text = queries_test.get(qid)
                if query_text is None:
                    print(f"Warning: Query ID {qid} not found in queries_test dictionary. Skipping.")
                    continue # Skip if query text isn't available
                query_length = len(query_text.split())

                # Select the appropriate model
                if query_length <= t1:
                    selected_model = short_model
                elif t1 < query_length <= t2:
                    selected_model = medium_model
                else: # query_length > t2
                    selected_model = long_model

                # Prepare inputs for the single item (add batch dimension)
                anchor_inputs = batch["anchor_input_ids"][i:i+1].to(device)
                anchor_masks = batch["anchor_attention_mask"][i:i+1].to(device)
                positive_inputs = batch["positive_input_ids"][i:i+1].to(device)
                positive_masks = batch["positive_attention_mask"][i:i+1].to(device)
                negative_inputs = batch["negative_input_ids"][i:i+1].to(device)
                negative_masks = batch["negative_attention_mask"][i:i+1].to(device)

                # Get embeddings from the selected model
                anchor_embeddings = selected_model(anchor_inputs, anchor_masks)
                positive_embeddings = selected_model(positive_inputs, positive_masks)
                negative_embeddings = selected_model(negative_inputs, negative_masks)

                # Calculate distances (assuming L2 norm, negate for score)
                pos_distance = torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=1)
                neg_distance = torch.norm(anchor_embeddings - negative_embeddings, p=2, dim=1)

                pos_score = -pos_distance.item()
                neg_score = -neg_distance.item()

                # Build the run dictionary
                if qid not in run:
                    run[qid] = {}

                # Ensure we don't overwrite potentially better scores if a doc appears multiple times
                run[qid][pos_did] = max(run[qid].get(pos_did, -float('inf')), pos_score)
                run[qid][neg_did] = max(run[qid].get(neg_did, -float('inf')), neg_score)


    # Calculate metrics
    metrics = [
        nDCG @ 10, nDCG @ 100,
        AP @ 10, AP @ 100,
        P @ 10, R @ 10,
        P @ 100, R @ 100,
        RR
    ]

    print("\nCalculating aggregate metrics...")
    metric_scores = calc_aggregate(metrics, qrels, run)
    print("Metrics calculation complete.")

    return metric_scores



def evaluate_weighted_average_ensemble(models, # List of models [short, medium, long]
                                       weights_config, # Dict defining weights per category
                                       t1, t2, # Query length thresholds
                                       test_loader, device, qrels, queries_test):
    """
    Evaluates using weighted score averaging ensemble based on query length.

    Args:
        models: List containing the short, medium, and long models.
                Order should match the weights_config categories.
        weights_config: Dictionary defining the weights. Example:
                        {
                            'short': [0.6, 0.2, 0.2], # Weights for [short, medium, long] models when query is short
                            'medium': [0.2, 0.6, 0.2],# Weights when query is medium
                            'long': [0.2, 0.2, 0.6]  # Weights when query is long
                        }
        t1: Upper bound length for short queries (inclusive).
        t2: Upper bound length for medium queries (inclusive).
        test_loader: DataLoader for the test set.
        device: The device to run inference on (e.g., 'cuda').
        qrels: Ground truth relevance judgments.
        queries_test: Dictionary mapping qid to query text for the test set.

    Returns:
        Dictionary of metric scores.
    """
    ft_models = [models['short_queries.pth'], models['medium_queries.pth'], models['long_queries.pth']]

    for model in ft_models:
        model.eval()
        model.to(device)
    
    num_models = len(ft_models)

    run = {}  # Format: {qid: {doc_id: score}}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Weighted Avg Ensemble"):
            qids = batch["qid"]
            pos_dids = batch["pos_did"]
            neg_dids = batch["neg_did"]

            # --- Get embeddings/distances from ALL models for the batch ---
            all_pos_distances = []
            all_neg_distances = []

            # Prepare inputs once per batch
            anchor_inputs = batch["anchor_input_ids"].to(device)
            anchor_masks = batch["anchor_attention_mask"].to(device)
            positive_inputs = batch["positive_input_ids"].to(device)
            positive_masks = batch["positive_attention_mask"].to(device)
            negative_inputs = batch["negative_input_ids"].to(device)
            negative_masks = batch["negative_attention_mask"].to(device)

            for model in ft_models:
                anchor_embeddings = model(anchor_inputs, anchor_masks)
                positive_embeddings = model(positive_inputs, positive_masks)
                negative_embeddings = model(negative_inputs, negative_masks)

                pos_distances = torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=1)
                neg_distances = torch.norm(anchor_embeddings - negative_embeddings, p=2, dim=1)
                all_pos_distances.append(pos_distances)
                all_neg_distances.append(neg_distances)

            # --- Combine scores based on query length for each item ---
            for i in range(len(qids)):
                qid = qids[i]
                pos_did = pos_dids[i]
                neg_did = neg_dids[i]

                # Get query text and length
                query_text = queries_test.get(qid)
                if query_text is None:
                    print(f"Warning: Query ID {qid} not found in queries_test dictionary. Skipping.")
                    continue
                query_length = len(query_text.split())

                # Determine weights based on query length category
                if query_length <= t1:
                    current_weights = weights_config['short']
                elif t1 < query_length <= t2:
                    current_weights = weights_config['medium']
                else: # query_length > t2
                    current_weights = weights_config['long']

                # Ensure weights match number of models
                if len(current_weights) != num_models:
                    raise ValueError(f"Number of weights ({len(current_weights)}) does not match number of models ({num_models})")

                # Calculate weighted score (negative distance)
                weighted_pos_score = 0.0
                weighted_neg_score = 0.0
                for m in range(num_models):
                    # Score is negative distance
                    weighted_pos_score += current_weights[m] * (-all_pos_distances[m][i].item())
                    weighted_neg_score += current_weights[m] * (-all_neg_distances[m][i].item())

                # Build the run dictionary
                if qid not in run:
                    run[qid] = {}

                # Ensure we don't overwrite potentially better scores
                run[qid][pos_did] = max(run[qid].get(pos_did, -float('inf')), weighted_pos_score)
                run[qid][neg_did] = max(run[qid].get(neg_did, -float('inf')), weighted_neg_score)


    # Calculate metrics
    metrics = [
        nDCG @ 10, nDCG @ 100,
        AP @ 10, AP @ 100,
        P @ 10, R @ 10,
        P @ 100, R @ 100,
        RR
    ]

    print("\nCalculating aggregate metrics...")
    metric_scores = calc_aggregate(metrics, qrels, run)
    print("Metrics calculation complete.")

    return metric_scores