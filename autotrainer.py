import os
import logging
import torch
from torch.optim import AdamW
from ir_measures import nDCG, P, R, RR
from IRutils import models, train, inference
from IRutils.load_data import load, preprocess


def run(model_name, dataset_name, length_setting):
    metrics = [
        nDCG @ 3, nDCG @ 5, nDCG @ 10, # Added nDCG@3
        RR,
        P @ 1, P @ 3, P @ 5,
        R @ 1, R @ 3, R @ 5, R @ 10    # Added R@1, R@3
    ]

    logging.disable(logging.WARNING)

    max_len_doc = 512  # max token length
    random_state = 42

    train_available, docs, queries, qrels, docs_test, queries_test, qrels_test = load(dataset_name)
    print('Loading complete!')

    if train_available:
        train_loader, val_loader, test_loader, split_queries_test, split_qrels_test, _, _ = preprocess(queries, docs, qrels,
                                                                                                 model_name,
                                                                                                 length_setting,
                                                                                                 train_available,
                                                                                                 queries_test=queries_test,
                                                                                                 docs_test=docs_test,
                                                                                                 qrels_test=qrels_test,
                                                                                                 max_len_doc=max_len_doc,
                                                                                                 random_state=random_state)
    else:
        train_loader, val_loader, test_loader, split_queries_test, split_qrels_test, _, _ = preprocess(queries, docs, qrels,
                                                                                                 model_name,
                                                                                                 length_setting,
                                                                                                 train_available,
                                                                                                 max_len_doc=max_len_doc,
                                                                                                 random_state=random_state)

    print('Preprocessing complete!')

    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model = models.TripletRankerModel(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Define model save dir
    os.makedirs(f'models/{model_name}/{dataset_name}', exist_ok=True)
    model_path = os.path.join(os.getcwd(), f'models/{model_name}/{dataset_name}/{length_setting}_queries.pth')

    print(f'Checking model path for existing model...: {model_path}')
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Train the model
        model = train.train_triplet_ranker(model, train_loader, val_loader, optimizer, device, model_path)
        # model = train.train_triplet_ranker_amp(model, train_loader, val_loader, optimizer, device, model_path) # amp enabled

    # Example usage (replace with your data and model)
    if train_available:
        metric_scores = inference.evaluate(model, test_loader, device, qrels_test)
    else:
        metric_scores = inference.evaluate(model, test_loader, device, split_qrels_test)

    for metric in metrics:
        print(f'Metric {metric} score: {metric_scores[metric]:.4f}')

    save_dir = f"results/{model_name}/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{length_setting}_queries.txt')

    inference.write_results(metric_scores, save_path, model_name, dataset_name, length_setting)


if __name__ == "__main__":
    run_models = ['distilroberta-base']
    run_datasets = ['fiqa', 'quora']
    run_length_settings = ['short', 'medium', 'long', 'full']

    # 24 runs
    for model in run_models:
        for dataset in run_datasets:
            for length in run_length_settings:
                print(f'Now training model {model} on dataset {dataset} for {length} query lengths...')
                run(model, dataset, length)

