# Import necessary modules from the previously provided code
# Assuming you have the previously defined classes: Config, DataManager, Trainer, etc.
import contextlib
import time

import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import gc  # Garbage collection for better memory management

from src.utils.create_datasets import DataManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(
            f"Available GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB used, {torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved")

    # Define configuration optimized for 8GB VRAM
    config = Config(
        model_name="distilbert-base-uncased",  # Light but effective model
        max_length=128,  # Reduced for memory efficiency
        batch_size=8,  # Small batch size for 8GB VRAM
        epochs=3,  # Increase for better results if time permits
        learning_rate=2e-5,
        warmup_steps=100,
        data_path="./data",
        save_dir="./models/distilbert",
        seed=42,
        eval_batch_size=16  # Larger for evaluation (no gradients stored)
    )

    # Enable gradient accumulation to simulate larger batch sizes
    gradient_accumulation_steps = 4  # Effective batch size = batch_size * gradient_accumulation_steps
    logger.info(
        f"Gradient accumulation steps: {gradient_accumulation_steps} (effective batch size: {config.batch_size * gradient_accumulation_steps})")

    # Create data manager and trainer
    data_manager = DataManager(config)
    trainer = Trainer(config)

    # Load corpus - contains the actual document text
    corpus_path = os.path.join(config.data_path, 'corpus/msmarco/corpus_path.txt')
    corpus = data_manager.load_corpus(corpus_path)

    # Train Model 1: Full dataset
    logger.info("====== Training Model 1: Full Dataset ======")
    train_corpus, train_query_categories = data_manager.get_query_length_datasets(split='train')
    val_corpus, val_query_categories = data_manager.get_query_length_datasets(split='val')

    # Combine all query categories for full model
    all_queries = pd.concat([cat_data['queries'] for cat_data in train_query_categories.values()])
    all_qrels = pd.concat([cat_data['qrels'] for cat_data in train_query_categories.values()])
    all_val_queries = pd.concat([cat_data['queries'] for cat_data in val_query_categories.values()])
    all_val_qrels = pd.concat([cat_data['qrels'] for cat_data in val_query_categories.values()])

    # Create dataloaders with smaller batch sizes and more workers
    full_train_dataset = RankingDataset(
        all_queries, train_corpus, all_qrels, trainer.tokenizer, config.max_length
    )
    full_val_dataset = RankingDataset(
        all_val_queries, val_corpus, all_val_qrels, trainer.tokenizer, config.max_length
    )

    full_train_dataloader = DataLoader(
        full_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    full_val_dataloader = DataLoader(
        full_val_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Train with gradient accumulation
    full_model = train_model_with_accumulation(
        trainer, full_train_dataloader, full_val_dataloader,
        gradient_accumulation_steps, model_suffix="_full"
    )

    # Clear memory before training next model
    del full_model, full_train_dataset, full_val_dataset
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Full dataset model training completed and memory cleared")

    # Train models for each query length category
    for category in ['short', 'medium', 'long']:
        logger.info(f"====== Training Model for {category.upper()} queries ======")

        # Create dataloaders for this category
        cat_train_dataset = RankingDataset(
            train_query_categories[category]['queries'],
            train_corpus,
            train_query_categories[category]['qrels'],
            trainer.tokenizer,
            config.max_length
        )

        # Use the same category for validation
        cat_val_dataset = RankingDataset(
            val_query_categories[category]['queries'],
            val_corpus,
            val_query_categories[category]['qrels'],
            trainer.tokenizer,
            config.max_length
        )

        cat_train_dataloader = DataLoader(
            cat_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        cat_val_dataloader = DataLoader(
            cat_val_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        # Train model for this category with gradient accumulation
        cat_model = train_model_with_accumulation(
            trainer, cat_train_dataloader, cat_val_dataloader,
            gradient_accumulation_steps, model_suffix=f"_{category}"
        )

        # Clear memory before training next model
        del cat_model, cat_train_dataset, cat_val_dataset
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"{category.capitalize()} query model training completed and memory cleared")


def train_model_with_accumulation(trainer, train_dataloader, val_dataloader, accumulation_steps, model_suffix=""):
    """
    Train model with gradient accumulation for better memory efficiency
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = trainer._create_model()
    model.to(device)

    # Enable mixed precision training for better memory efficiency and speed
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=trainer.config.learning_rate, weight_decay=trainer.config.weight_decay)

    total_steps = len(train_dataloader) * trainer.config.epochs // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=trainer.config.warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(trainer.config.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{trainer.config.epochs}")

        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()  # Zero gradients at the beginning

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                            desc=f"Training epoch {epoch + 1}")

        for step, batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass with mixed precision
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask)
                    loss = torch.nn.functional.mse_loss(outputs, labels)
                    loss = loss / accumulation_steps  # Normalize loss

                # Backward pass with scaling
                scaler.scale(loss).backward()

                # Accumulate gradients
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    # Unscale before clipping to avoid underflow
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # Optimizer step with scaling
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                # Standard training path without mixed precision
                outputs = model(input_ids, attention_mask)
                loss = torch.nn.functional.mse_loss(outputs, labels)
                loss = loss / accumulation_steps  # Normalize loss

                loss.backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # Track loss
            train_loss += loss.item() * accumulation_steps

            # Update progress bar
            progress_bar.set_postfix({'loss': train_loss / (step + 1)})

            # Monitor GPU memory usage
            if step % 100 == 0 and torch.cuda.is_available():
                logger.debug(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB used")

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logger.info(f"Average training loss: {avg_train_loss}")

        # Validation
        if val_dataloader:
            # Clear memory before validation
            torch.cuda.empty_cache()

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Validating epoch {epoch + 1}"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(input_ids, attention_mask)
                    loss = torch.nn.functional.mse_loss(outputs, labels)

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            logger.info(f"Average validation loss: {avg_val_loss}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                trainer.save_model(model, f"best{model_suffix}")
                logger.info(f"Saved best model with validation loss: {best_val_loss}")

        # Save model after each epoch
        trainer.save_model(model, f"epoch_{epoch + 1}{model_suffix}")

    # Save final model
    trainer.save_model(model, f"final{model_suffix}")

    return model


def evaluate_models():
    """
    Evaluate all trained models on the test set with GPU optimization
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating models using {device}")

    config = Config.load("./models/distilbert/config_final_full.json")
    # Adjust batch size for evaluation (can be larger since no gradients are stored)
    config.eval_batch_size = 32  # Increase if memory allows

    data_manager = DataManager(config)
    trainer = Trainer(config)

    # Load test data
    test_corpus, test_query_categories = data_manager.get_query_length_datasets(split='test')

    # Combine all test queries for overall evaluation
    all_test_queries = pd.concat([cat_data['queries'] for cat_data in test_query_categories.values()])
    all_test_qrels = pd.concat([cat_data['qrels'] for cat_data in test_query_categories.values()])

    # Create test dataloader
    test_dataset = RankingDataset(
        all_test_queries, test_corpus, all_test_qrels, trainer.tokenizer, config.max_length
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Create category-specific test dataloaders
    cat_test_dataloaders = {}
    for category in ['short', 'medium', 'long']:
        cat_test_dataset = RankingDataset(
            test_query_categories[category]['queries'],
            test_corpus,
            test_query_categories[category]['qrels'],
            trainer.tokenizer,
            config.max_length
        )
        cat_test_dataloaders[category] = DataLoader(
            cat_test_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

    # Load and evaluate models one by one to save memory
    model_names = ['full', 'short', 'medium', 'long']
    results = {}

    for model_name in model_names:
        # Clean up memory before loading new model
        torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"Loading {model_name} model")
        model = trainer.load_model(f"final_{model_name}")
        model.to(device)
        model.eval()

        # Evaluate on full test set
        logger.info(f"Evaluating {model_name} model on full test set")
        results_df = trainer.evaluate_model(model, test_dataloader)
        metrics = trainer.evaluate_metrics(results_df)
        results[f"{model_name}_full"] = metrics

        # Evaluate on category-specific test sets
        for category, dataloader in cat_test_dataloaders.items():
            logger.info(f"Evaluating {model_name} model on {category} test set")
            results_df = trainer.evaluate_model(model, dataloader)
            metrics = trainer.evaluate_metrics(results_df)
            results[f"{model_name}_{category}"] = metrics

        # Free up memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Create results dataframe and save to CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv("./models/distilbert/evaluation_results.csv")
    logger.info(f"Evaluation results saved to ./models/distilbert/evaluation_results.csv")

    return results_df


def dynamic_query_routing():
    """
    Implementation of dynamic query routing based on query length
    with efficient GPU usage for 8GB VRAM
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing dynamic query router using {device}")

    config = Config.load("./models/distilbert/config_final_full.json")

    # We'll load models on-demand to save memory
    trainer = Trainer(config)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.save_dir, "tokenizer_final_full"))

    # Define model paths for lazy loading
    model_paths = {
        'short': os.path.join(config.save_dir, "model_final_short"),
        'medium': os.path.join(config.save_dir, "model_final_medium"),
        'long': os.path.join(config.save_dir, "model_final_long")
    }

    # Keep track of currently loaded model to avoid reloading the same model
    currently_loaded = {'category': None, 'model': None}

    # Function to route a query to the appropriate model
    def route_query(query_text, documents, top_k=10):
        """
        Route query to appropriate model based on length and rank documents

        Args:
            query_text (str): The query text
            documents (list): List of document dictionaries with at least a 'docno' and 'text' field
            top_k (int): Number of top documents to return

        Returns:
            list: Ranked document IDs
        """
        # Determine query length category
        query_length = len(query_text.split())

        # Define thresholds for categories (adjust based on your data)
        if query_length <= 3:
            category = 'short'
        elif query_length <= 7:
            category = 'medium'
        else:
            category = 'long'

        logger.info(f"Query '{query_text}' classified as {category} with {query_length} words")

        # Load appropriate model if not already loaded
        if currently_loaded['category'] != category:
            # Clear previous model from memory if one exists
            if currently_loaded['model'] is not None:
                del currently_loaded['model']
                torch.cuda.empty_cache()
                gc.collect()

            # Create a new model instance
            model = NeuralRanker(config.model_name)
            model.load_state_dict(torch.load(model_paths[category], map_location=device))
            model.to(device)
            model.eval()

            # Update currently loaded model
            currently_loaded['category'] = category
            currently_loaded['model'] = model
            logger.info(f"Loaded {category} model for dynamic routing")

        # Rank documents using the loaded model
        model = currently_loaded['model']
        scores = []

        # Process in batches to avoid memory issues
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]

                # Tokenize query-document pairs
                batch_inputs = []
                batch_doc_ids = []

                for doc in batch_docs:
                    encoded = tokenizer(
                        query_text,
                        doc['text'],
                        truncation='longest_first',
                        max_length=config.max_length,
                        padding='max_length',
                        return_tensors='pt'
                    )
                    batch_inputs.append({
                        'input_ids': encoded['input_ids'].squeeze(0),
                        'attention_mask': encoded['attention_mask'].squeeze(0)
                    })
                    batch_doc_ids.append(doc['docno'])

                # Create tensors for batch processing
                input_ids = torch.stack([inputs['input_ids'] for inputs in batch_inputs]).to(device)
                attention_mask = torch.stack([inputs['attention_mask'] for inputs in batch_inputs]).to(device)

                # Get scores
                with torch.cuda.amp.autocast() if torch.cuda.is_available() else contextlib.nullcontext():
                    batch_scores = model(input_ids, attention_mask)

                # Add scores to list
                scores.extend([(doc_id, score.item()) for doc_id, score in zip(batch_doc_ids, batch_scores)])

        # Sort documents by score in descending order
        ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)

        # Return top-k document IDs
        return [doc_id for doc_id, _ in ranked_docs[:top_k]]

    return route_query


# Example function to perform inference with the router
def perform_inference(test_queries, corpus_df, router_func):
    """
    Run inference for test queries using the dynamic router

    Args:
        test_queries (list): List of query strings
        corpus_df (DataFrame): DataFrame containing document corpus
        router_func (function): The dynamic query routing function
    """
    results = {}

    for query in test_queries:
        # Prepare documents (convert DataFrame to list of dicts)
        documents = [
            {'docno': row.name, 'text': row['passage']}
            for _, row in corpus_df.iterrows()
        ]

        # Use the router to get ranked documents
        start_time = time.time()
        ranked_doc_ids = router_func(query, documents, top_k=10)
        inference_time = time.time() - start_time

        results[query] = {
            'ranked_docs': ranked_doc_ids,
            'inference_time': inference_time,
            'category': query_length_category(query)
        }

        logger.info(f"Processed query: '{query}'")
        logger.info(f"Query category: {query_length_category(query)}")
        logger.info(f"Inference time: {inference_time:.4f} seconds")
        logger.info(f"Top ranked document: {ranked_doc_ids[0] if ranked_doc_ids else 'None'}")
        logger.info("-" * 50)

    return results


def query_length_category(query_text):
    """Helper function to determine query length category"""
    query_length = len(query_text.split())
    if query_length <= 3:
        return 'short'
    elif query_length <= 7:
        return 'medium'
    else:
        return 'long'


# Complete main function with GPU optimizations
if __name__ == "__main__":
    # Add file handler for logging
    file_handler = logging.FileHandler('training.log')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Print GPU info
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available. Training will be slow on CPU!")

    # Set torch deterministic for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enable memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        # Train all models
        logger.info("Starting training process")
        main()

        # Report peak memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"Peak GPU memory usage during training: {peak_memory:.2f} GB")

        # Evaluate all models
        logger.info("Starting evaluation process")
        eval_results = evaluate_models()
        print("Evaluation results summary:")
        print(eval_results)

        # Set up dynamic query routing
        logger.info("Setting up dynamic query router")
        router = dynamic_query_routing()

        # Example of using the router (if needed)
        test_queries = [
            "what is respite",  # short
            "how to repair a leaking faucet",  # medium
            "what causes a prickly itchy feeling all over the body and how to treat it"  # long
        ]

        logger.info("Dynamic query router created successfully")

        logger.info("All processes completed successfully")
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        logger.exception("Exception details:")

        # Report memory usage when exception occurs
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1e9
            logger.error(f"GPU memory at exception: {current_memory:.2f} GB")