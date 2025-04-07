import numpy as np
import torch
from tqdm import tqdm
from ir_measures import *
from scipy import stats  # For t-test
import pandas as pd


def write_results(metric_scores, save_path, model_name, dataset_name, length_setting):
    """
    Saves evaluation results to a file, reporting the specified metrics.

    Args:
        metric_scores: A dictionary-like object mapping ir_measures metric objects
                       to their calculated scores. Must include keys for the metrics
                       defined in `required_metrics`.
        save_path: Path to the file where results will be saved.
        model_name: Name of the model being evaluated.
        dataset_name: Name of the dataset used.
        length_setting: Description of the query length focus or model variant
                        (e.g., "short", "all", "baseline").
    """
    # Define the required metric objects for key access and reporting
    # These MUST match the metrics calculated in your evaluate function
    required_metrics = {
        'nDCG@10': nDCG @ 100,
        'RR': RR,
        'R@100': R @ 100,
    }

    # Check if all required metric scores are present in the input dict
    scores = {}
    missing_metrics = []
    found_metric_names = []
    for name, metric_obj in required_metrics.items():
        if metric_obj not in metric_scores:
            missing_metrics.append(name)
        else:
            scores[name] = metric_scores[metric_obj]
            found_metric_names.append(name)

    if missing_metrics:
        print(f"Metric scores not found for: {', '.join(missing_metrics)}. "
                     f"Metrics found were: {found_metric_names}. "
                     f"Ensure your evaluate function calculates all required metrics: "
                     f"{list(required_metrics.keys())}")
        # Decide if you want to proceed with partial results or exit
        # For now, we'll proceed with what we have, but log the error.
        # return # Uncomment this to exit if any required metric is missing

    print(f"Writing results for metrics: {list(scores.keys())} to {save_path}")

    # Save results to a file
    try:
        with open(save_path, "w") as f:
            f.write(f"Evaluation Results for {model_name} model ({length_setting}) on {dataset_name} dataset:\n")
            f.write("----------------------------------------------------\n")

            # --- Ranking/Overall Quality ---
            if 'nDCG@100' in scores:
                f.write(f"nDCG@100: {scores['nDCG@100']:.4f}  (Quality of Top 100 Ranking)\n")
            if 'RR' in scores:
                f.write(f"RR:  {scores['RR']:.4f}  (Mean Reciprocal Rank)\n")
            f.write("\n")

            # --- Recall Metric ---
            if 'R@100' in scores:
                f.write(f"R@100:   {scores['R@100']:.4f}  (Recall within Top 100)\n")
            f.write("\n")


            f.write("----------------------------------------------------\n")
            f.write("\n")
            f.write("Explanation of reported metrics:\n")
            f.write(
                "  nDCG@k: Measures ranking quality, rewarding highly relevant documents found earlier.\n"
                "          Normalized discount cumulative gain. Good overall top-k ranking indicator.\n"
            )
            f.write(
                "  RR:     The [Mean] Reciprocal Rank ([M]RR) is a precision-focused measure that scores\n"
                "          based on the reciprocal of the rank of the highest-scoring relevance document.\n"
            )
            f.write(
                "  R@k:    Recall@k. Fraction of *known* relevant documents found in the top k results.\n"
                "          Measures coverage within a practical top set (@100).\n"
            )

    except IOError as e:
        print(f"Failed to write results to {save_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during result writing: {e}")


def get_per_query_metrics(metrics, qrels, run):
    """Calculates metrics for each query individually."""
    per_query_results = {} # {qid: {metric_name: score}}
    # ir_measures.iter_calc returns an iterator of measurement objects
    # Each measurement has qid, metric, value
    for measurement in iter_calc(metrics, qrels, run):
        qid = measurement.query_id
        metric_name = str(measurement.measure) # Get a string representation
        score = measurement.value
        if qid not in per_query_results:
            per_query_results[qid] = {}
        per_query_results[qid][metric_name] = score
    return per_query_results


def evaluate(model, test_loader, device, qrels):
    """
    Modified to return both aggregate and per-query scores
    """
    model.eval()
    model.to(device)
    run = {}  # Format: {qid: {doc_id: score}}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            qids = batch["qid"]
            doc_ids = batch["doc_id"]

            # Process embeddings and calculate distances
            query_inputs = batch["query_input_ids"].to(device)
            query_masks = batch["query_attention_mask"].to(device)
            doc_inputs = batch["doc_input_ids"].to(device)
            doc_masks = batch["doc_attention_mask"].to(device)

            query_embeddings = model(query_inputs, query_masks)
            doc_embeddings = model(doc_inputs, doc_masks)

            l2_distance = torch.norm(query_embeddings - doc_embeddings, p=2, dim=1)

            # Build the run dictionary
            i = 0
            while i < len(qids):
                qid = qids[i]
                doc_id = doc_ids[i]

                score = -l2_distance[i].item()

                if qid not in run:
                    run[qid] = {}

                # Add scores directly (no list of dicts)
                run[qid][doc_id] = score

                i += 1

    # Define metrics
    metrics = [
        nDCG @ 100,  # Quality of top 100 results (standard)
        R @ 100,  # Recall within top 100
        RR
    ]

    # Calculate aggregate metrics
    metric_scores = calc_aggregate(metrics, qrels, run)

    # Calculate per-query metrics for statistical tests
    per_query_scores = get_per_query_metrics(metrics, qrels, run)

    return metric_scores, per_query_scores, run


def evaluate_average_ensemble(models, test_loader, device, qrels):
    """
    Modified to return both aggregate and per-query scores
    """
    run = {}  # Format: {qid: {doc_id: score}}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            qids = batch["qid"]
            doc_ids = batch["doc_id"]

            # Process embeddings and calculate distances
            query_inputs = batch["query_input_ids"].to(device)
            query_masks = batch["query_attention_mask"].to(device)
            doc_inputs = batch["doc_input_ids"].to(device)
            doc_masks = batch["doc_attention_mask"].to(device)

            l2_distances = []

            for model in models.values():
                model.eval()
                model.to(device)

                query_embeddings = model(query_inputs, query_masks)
                doc_embeddings = model(doc_inputs, doc_masks)

                l2_distance = torch.norm(query_embeddings - doc_embeddings, p=2, dim=1)

                l2_distances.append(l2_distance)

            # Compute the average distances across all three models
            final_doc_distances = torch.stack(l2_distances).mean(dim=0)  # Average over models

            # Build the run dictionary
            i = 0
            while i < len(qids):
                qid = qids[i]
                doc_id = doc_ids[i]

                score = -final_doc_distances[i].item()

                if qid not in run:
                    run[qid] = {}

                # Add scores directly (no list of dicts)
                run[qid][doc_id] = score

                i += 1

    # Define metrics
    metrics = [
        nDCG @ 100,  # Quality of top 100 results (standard)
        R @ 100,  # Recall within top 100
        RR
    ]

    # Calculate aggregate metrics
    metric_scores = calc_aggregate(metrics, qrels, run)

    # Calculate per-query metrics for statistical tests
    per_query_scores = get_per_query_metrics(metrics, qrels, run)

    return metric_scores, per_query_scores, run


def evaluate_conditional_ensemble(models, t1, t2, test_loader, device, qrels, queries_test):
    """
    Modified to return both aggregate and per-query scores
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
            doc_ids = batch["doc_id"]

            # We need to process each item individually because model selection depends on query length
            for i in range(len(qids)):
                qid = qids[i]
                doc_id = doc_ids[i]

                # Get query text and length
                query_text = queries_test.get(qid)
                if query_text is None:
                    print(f"Warning: Query ID {qid} not found in queries_test dictionary. Skipping.")
                    continue  # Skip if query text isn't available
                query_length = len(query_text.split())

                # Select the appropriate model
                if query_length <= t1:
                    selected_model = short_model
                elif t1 < query_length <= t2:
                    selected_model = medium_model
                else:  # query_length > t2
                    selected_model = long_model

                # Prepare inputs for the single item (add batch dimension)
                query_inputs = batch["query_input_ids"][i:i + 1].to(device)
                query_masks = batch["query_attention_mask"][i:i + 1].to(device)
                doc_inputs = batch["doc_input_ids"][i:i + 1].to(device)
                doc_masks = batch["doc_attention_mask"][i:i + 1].to(device)

                # Get embeddings from the selected model
                query_embeddings = selected_model(query_inputs, query_masks)
                doc_embeddings = selected_model(doc_inputs, doc_masks)

                # Calculate distances (assuming L2 norm, negate for score)
                l2_distance = torch.norm(query_embeddings - doc_embeddings, p=2, dim=1)

                score = -l2_distance.item()

                # Build the run dictionary
                if qid not in run:
                    run[qid] = {}

                # Ensure we don't overwrite potentially better scores if a doc appears multiple times
                run[qid][doc_id] = max(run[qid].get(doc_id, -float('inf')), score)

    # Define metrics
    metrics = [
        nDCG @ 100,  # Quality of top 100 results (standard)
        R @ 100,  # Recall within top 100
        RR
    ]

    # Calculate aggregate metrics
    print("\nCalculating aggregate metrics...")
    metric_scores = calc_aggregate(metrics, qrels, run)

    # Calculate per-query metrics for statistical tests
    per_query_scores = get_per_query_metrics(metrics, qrels, run)

    print("Metrics calculation complete.")

    return metric_scores, per_query_scores, run


def evaluate_weighted_average_ensemble(models, weights_config, t1, t2, test_loader, device, qrels, queries_test):
    """
    Modified to return both aggregate and per-query scores
    """
    ft_models = [models['short'], models['medium'], models['long']]

    for model in ft_models:
        model.eval()
        model.to(device)

    num_models = len(ft_models)

    run = {}  # Format: {qid: {doc_id: score}}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Weighted Avg Ensemble"):
            qids = batch["qid"]
            doc_ids = batch["doc_id"]

            # --- Get embeddings/distances from ALL models for the batch ---
            all_doc_distances = []

            # Process embeddings and calculate distances
            query_inputs = batch["query_input_ids"].to(device)
            query_masks = batch["query_attention_mask"].to(device)
            doc_inputs = batch["doc_input_ids"].to(device)
            doc_masks = batch["doc_attention_mask"].to(device)

            for model in ft_models:
                query_embeddings = model(query_inputs, query_masks)
                doc_embeddings = model(doc_inputs, doc_masks)

                l2_distance = torch.norm(query_embeddings - doc_embeddings, p=2, dim=1)
                all_doc_distances.append(l2_distance)

            # --- Combine scores based on query length for each item ---
            for i in range(len(qids)):
                qid = qids[i]
                doc_id = doc_ids[i]

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
                else:  # query_length > t2
                    current_weights = weights_config['long']

                # Ensure weights match number of models
                if len(current_weights) != num_models:
                    raise ValueError(
                        f"Number of weights ({len(current_weights)}) does not match number of models ({num_models})")

                # Calculate weighted score (negative distance)
                weighted_score = 0.0
                for m in range(num_models):
                    # Score is negative distance
                    weighted_score += current_weights[m] * (-all_doc_distances[m][i].item())

                # Build the run dictionary
                if qid not in run:
                    run[qid] = {}

                # Ensure we don't overwrite potentially better scores
                run[qid][doc_id] = max(run[qid].get(doc_id, -float('inf')), weighted_score)

    # Define metrics
    metrics = [
        nDCG @ 100,  # Quality of top 100 results (standard)
        R @ 100,  # Recall within top 100
        RR
    ]

    # Calculate aggregate metrics
    print("\nCalculating aggregate metrics...")
    metric_scores = calc_aggregate(metrics, qrels, run)

    # Calculate per-query metrics for statistical tests
    per_query_scores = get_per_query_metrics(metrics, qrels, run)

    print("Metrics calculation complete.")

    return metric_scores, per_query_scores, run


def perform_ttest(baseline_per_query, model_per_query, metric, alpha=0.05):
    """
    Performs a paired t-test between baseline and model results.

    Args:
        baseline_per_query: Per-query results from the baseline model
        model_per_query: Per-query results from the comparison model
        metric: Metric object to test (e.g., nDCG @ 10)
        alpha: Significance level

    Returns:
        Dictionary with t-statistic, p-value, and significance result
    """
    # Extract per-query scores for the specified metric
    baseline_scores = []
    model_scores = []

    # Get common query IDs
    common_qids = set(baseline_per_query.keys()) & set(model_per_query.keys())

    for qid in common_qids:
        # Find the metric in baseline_per_query[qid]
        for entry in baseline_per_query[qid]:
            if entry.measure == metric:
                baseline_scores.append(entry.value)
                break

        # Find the metric in model_per_query[qid]
        for entry in model_per_query[qid]:
            if entry.measure == metric:
                model_scores.append(entry.value)
                break

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(model_scores, baseline_scores)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'improvement': np.mean(model_scores) > np.mean(baseline_scores)
    }


def compare_models_with_ttest(baseline_results, models_results, metrics, model_names):
    """
    Compares each model to the baseline using t-tests for multiple metrics.

    Args:
        baseline_results: Tuple with (aggregate_scores, per_query_scores, run) from baseline
        models_results: List of tuples with results for each model being compared
        metrics: List of metric objects to test
        model_names: List of names for each model

    Returns:
        DataFrame with comparison results
    """
    _, baseline_per_query, _ = baseline_results

    results = []

    for i, (_, model_per_query, _) in enumerate(models_results):
        model_name = model_names[i]

        for metric in metrics:
            test_result = perform_ttest(baseline_per_query, model_per_query, metric)

            # Get mean scores for baseline and this model
            baseline_mean = np.mean([entry.value for qid in baseline_per_query
                                     for entry in baseline_per_query[qid] if entry.measure == metric])
            model_mean = np.mean([entry.value for qid in model_per_query
                                  for entry in model_per_query[qid] if entry.measure == metric])

            results.append({
                'Model': model_name,
                'Metric': str(metric),
                'Baseline Score': baseline_mean,
                'Model Score': model_mean,
                'Difference': model_mean - baseline_mean,
                'Improvement %': ((model_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0,
                'T-Statistic': test_result['t_statistic'],
                'P-Value': test_result['p_value'],
                'Significant': test_result['significant'],
                'Improvement': test_result['improvement']
            })

    return pd.DataFrame(results)


def write_ttest_results(ttest_df, save_path, baseline_name="Baseline"):
    """
    Saves t-test comparison results to a file.

    Args:
        ttest_df: DataFrame with t-test results
        save_path: Path to save the results
        baseline_name: Name of the baseline model for the report
    """
    with open(save_path, "w") as f:
        f.write(f"T-Test Comparison Results (compared to {baseline_name})\n")
        f.write("==================================================\n\n")

        # Group by model
        for model_name, group in ttest_df.groupby('Model'):
            f.write(f"Model: {model_name}\n")
            f.write("-" * (7 + len(model_name)) + "\n")

            for _, row in group.iterrows():
                metric = row['Metric']
                baseline = row['Baseline Score']
                model = row['Model Score']
                diff = row['Difference']
                pct = row['Improvement %']
                p_val = row['P-Value']
                sig = "✓" if row['Significant'] else "✗"
                imp = "+" if row['Improvement'] else "-"

                f.write(f"{metric}:\n")
                f.write(f"  {baseline_name}: {baseline:.4f}, {model_name}: {model:.4f}\n")
                f.write(f"  Diff: {diff:.4f} ({imp}{abs(pct):.2f}%)\n")
                f.write(f"  p-value: {p_val:.6f} {'(significant)' if row['Significant'] else ''}\n\n")

            f.write("\n")

        # Summary of significant improvements
        significant_improvements = ttest_df[(ttest_df['Significant'] == True) &
                                            (ttest_df['Improvement'] == True)]

        if not significant_improvements.empty:
            f.write("\nSummary of Significant Improvements:\n")
            f.write("==================================\n")

            for _, row in significant_improvements.iterrows():
                f.write(f"{row['Model']} significantly improves {row['Metric']} by {row['Improvement %']:.2f}%\n")
                f.write(f"  (p-value: {row['P-Value']:.6f})\n\n")

    print(f"T-test results written to {save_path}")