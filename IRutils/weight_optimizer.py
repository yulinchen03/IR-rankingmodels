import torch
import optuna
from ir_measures import calc_aggregate, nDCG, RR, P, R # Make sure these are imported
import numpy as np
from tqdm import tqdm # For progress bar inside objective if needed

def precompute_validation_scores(models, val_loader, device):
    """
    Pre-computes scores (negative distances) for validation data using all models.

    Args:
        models: Dictionary of loaded models {'short': model_s, 'medium': model_m, 'long': model_l}.
        val_loader: DataLoader for the validation set.
        device: The device to run inference on.

    Returns:
        Dictionary mapping qid to another dictionary:
        { qid: {
            'pos_did': pos_did,
            'neg_did': neg_did,
            'pos_scores': [score_short, score_medium, score_long], # Negative distances
            'neg_scores': [score_short, score_medium, score_long]  # Negative distances
          }
        }
        Note: Assumes val_loader provides one pos/neg pair per anchor.
              If multiple negatives exist per anchor in val_loader, this needs adjustment.
    """
    print("Pre-computing validation scores for all models...")
    val_scores = {}
    model_list = [models['short'], models['medium'], models['long']]

    # Set models to evaluation mode
    for model in model_list:
        model.eval()
        model.to(device)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Precomputing Val Scores"):
            qids = batch["qid"]
            pos_dids = batch["pos_did"]
            neg_dids = batch["neg_did"]

            # Prepare inputs once per batch
            anchor_inputs = batch["anchor_input_ids"].to(device)
            anchor_masks = batch["anchor_attention_mask"].to(device)
            positive_inputs = batch["positive_input_ids"].to(device)
            positive_masks = batch["positive_attention_mask"].to(device)
            negative_inputs = batch["negative_input_ids"].to(device)
            negative_masks = batch["negative_attention_mask"].to(device)

            batch_pos_scores = []
            batch_neg_scores = []

            for model in model_list:
                anchor_embeddings = model(anchor_inputs, anchor_masks)
                positive_embeddings = model(positive_inputs, positive_masks)
                negative_embeddings = model(negative_inputs, negative_masks)

                pos_distances = torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=1)
                neg_distances = torch.norm(anchor_embeddings - negative_embeddings, p=2, dim=1)

                # Store negative distances as scores
                batch_pos_scores.append(-pos_distances.cpu().numpy())
                batch_neg_scores.append(-neg_distances.cpu().numpy())

            # Transpose scores to group by item, not by model
            # Resulting shape: (batch_size, num_models)
            item_pos_scores = np.stack(batch_pos_scores, axis=1)
            item_neg_scores = np.stack(batch_neg_scores, axis=1)

            # Store results per query ID
            for i in range(len(qids)):
                qid = qids[i]
                # Simple storage assuming one pos/neg per anchor in val loader
                # If multiple instances for a qid appear, this might overwrite.
                # A more robust approach might store lists per qid if needed.
                val_scores[qid] = {
                    'pos_did': pos_dids[i],
                    'neg_did': neg_dids[i],
                    'pos_scores': item_pos_scores[i], # [score_s, score_m, score_l]
                    'neg_scores': item_neg_scores[i]  # [score_s, score_m, score_l]
                 }

    print("Pre-computation complete.")
    return val_scores



def find_optimal_weights_config(precomputed_scores, query_val, qrels_val, t1, t2, metric_to_optimize=nDCG@10, n_trials=100, random_state=42):
    """
    Uses Optuna to find the best weights_config dictionary by maximizing
    a metric on the entire validation set.

    Args:
        precomputed_scores (dict): Output from precompute_validation_scores.
        query_val (dict): Validation queries {qid: text}.
        qrels_val (dict): Validation qrels.
        t1 (int): Threshold between short and medium.
        t2 (int): Threshold between medium and long.
        metric_to_optimize (Measure): The ir_measures metric to maximize.
        n_trials (int): Number of optimization trials Optuna should run.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: The best weights_config dictionary found.
              {'short': [ws_s, ws_m, ws_l], 'medium': [wm_s, wm_m, wm_l], 'long': [wl_s, wl_m, wl_l]}
    """
    print(f"\nOptimizing full weights_config using {metric_to_optimize} on the validation set...")

    # Define the objective function for Optuna
    def objective(trial):
        # --- Suggest weights for each category ---
        # Weights for SHORT queries
        ws_s = trial.suggest_float('ws_s', 0.0, 1.0)
        ws_m = trial.suggest_float('ws_m', 0.0, 1.0 - ws_s)
        ws_l = max(0.0, 1.0 - ws_s - ws_m)
        weights_short = np.array([ws_s, ws_m, ws_l])
        if weights_short.sum() > 1e-6: weights_short /= weights_short.sum()
        else: weights_short = np.array([1/3]*3)

        # Weights for MEDIUM queries
        wm_s = trial.suggest_float('wm_s', 0.0, 1.0)
        wm_m = trial.suggest_float('wm_m', 0.0, 1.0 - wm_s)
        wm_l = max(0.0, 1.0 - wm_s - wm_m)
        weights_medium = np.array([wm_s, wm_m, wm_l])
        if weights_medium.sum() > 1e-6: weights_medium /= weights_medium.sum()
        else: weights_medium = np.array([1/3]*3)

        # Weights for LONG queries
        wl_s = trial.suggest_float('wl_s', 0.0, 1.0)
        wl_m = trial.suggest_float('wl_m', 0.0, 1.0 - wl_s)
        wl_l = max(0.0, 1.0 - wl_s - wl_m)
        weights_long = np.array([wl_s, wl_m, wl_l])
        if weights_long.sum() > 1e-6: weights_long /= weights_long.sum()
        else: weights_long = np.array([1/3]*3)

        current_weights_config = {
            'short': weights_short.tolist(),
            'medium': weights_medium.tolist(),
            'long': weights_long.tolist()
        }

        # --- Evaluate this config on the ENTIRE validation set ---
        run = {}
        for qid, data in precomputed_scores.items():
            if qid not in query_val: continue # Ensure query text is available

            query_text = query_val[qid]
            query_length = len(query_text.split())

            # Determine weights based on query length category
            if query_length <= t1:
                current_weights = current_weights_config['short']
            elif t1 < query_length <= t2:
                current_weights = current_weights_config['medium']
            else: # query_length > t2
                current_weights = current_weights_config['long']

            pos_scores_models = data['pos_scores'] # [score_s, score_m, score_l]
            neg_scores_models = data['neg_scores'] # [score_s, score_m, score_l]

            # Calculate weighted score (negative distance)
            # Ensure weights are numpy array for dot product
            weights_np = np.array(current_weights)
            weighted_pos_score = np.dot(weights_np, pos_scores_models)
            weighted_neg_score = np.dot(weights_np, neg_scores_models)

            if qid not in run:
                run[qid] = {}

            # Add scores (handle potential multiple entries per doc if val_loader structure changes)
            run[qid][data['pos_did']] = max(run[qid].get(data['pos_did'], -float('inf')), weighted_pos_score)
            run[qid][data['neg_did']] = max(run[qid].get(data['neg_did'], -float('inf')), weighted_neg_score)

        # Calculate the aggregate metric over the full validation qrels
        if not run or not qrels_val:
             # print("Warning: Empty run or qrels for validation set in objective. Returning 0.")
             return 0.0 # Can't calculate metric

        metric_scores = calc_aggregate([metric_to_optimize], qrels_val, run)

        # Optuna maximizes, so return the score directly
        return metric_scores.get(metric_to_optimize, 0.0) # Return 0 if metric calculation fails

    # Create and run the Optuna study
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=random_state), # Ensure reproducibility
                                pruner=optuna.pruners.MedianPruner())
    # Use progress bar for optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # --- Extract the best weights config ---
    best_params = study.best_params
    ws_s = best_params['ws_s']
    ws_m = best_params['ws_m']
    ws_l = max(0.0, 1.0 - ws_s - ws_m)
    best_weights_short = np.array([ws_s, ws_m, ws_l])
    if best_weights_short.sum() > 1e-6: best_weights_short /= best_weights_short.sum()
    else: best_weights_short = np.array([1/3]*3)

    wm_s = best_params['wm_s']
    wm_m = best_params['wm_m']
    wm_l = max(0.0, 1.0 - wm_s - wm_m)
    best_weights_medium = np.array([wm_s, wm_m, wm_l])
    if best_weights_medium.sum() > 1e-6: best_weights_medium /= best_weights_medium.sum()
    else: best_weights_medium = np.array([1/3]*3)

    wl_s = best_params['wl_s']
    wl_m = best_params['wl_m']
    wl_l = max(0.0, 1.0 - wl_s - wl_m)
    best_weights_long = np.array([wl_s, wl_m, wl_l])
    if best_weights_long.sum() > 1e-6: best_weights_long /= best_weights_long.sum()
    else: best_weights_long = np.array([1/3]*3)

    best_weights_config = {
        'short': best_weights_short.tolist(),
        'medium': best_weights_medium.tolist(),
        'long': best_weights_long.tolist()
    }

    print(f"Optimization complete. Best overall score ({metric_to_optimize}) on validation: {study.best_value:.4f}")
    return best_weights_config