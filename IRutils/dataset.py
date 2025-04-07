# --- START OF FILE dataset.py ---

import random
from typing import Dict, Union, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

from transformers import PreTrainedModel, PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# for train and val
class TripletRankingDataset(Dataset):
    def __init__(self, queries, documents, qrels, tokenizer, num_negatives, neg_sample_size=10, max_length=512, for_eval=False):
        self.triplets = []
        self.num_negatives = num_negatives
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.queries = queries
        self.documents = documents # Store documents ref

        # For each query, find positive documents and sample negatives
        for qid, doc_dict in tqdm(qrels.items(), desc="Building Triplets"):
            if qid not in queries:
                continue

            query = queries[qid]
            positive_docs_ids = [did for did, relevance in doc_dict.items() if relevance > 0 and did in self.documents]

            if not positive_docs_ids: # Skip queries with no relevant docs in the corpus
                continue

            # Handle negative sampling or extraction
            if self.num_negatives == 0: # Sample negatives
                 all_doc_ids = list(self.documents.keys())
                 negative_candidates = [did for did in all_doc_ids if did not in positive_docs_ids]
                 if not negative_candidates: # Should not happen often, but handle
                     continue
                 # Sample negatives per positive doc
                 for pos_did in positive_docs_ids:
                     if for_eval:
                        neg_dids = negative_candidates[:(100-len(positive_docs_ids))]  # sample a total of 100 documents from corpus for ranking
                     else:
                        neg_dids = random.sample(negative_candidates, min(neg_sample_size, len(negative_candidates)))  # sample 10 documents per positive doc for training
                     for neg_did in neg_dids:
                         self.triplets.append((qid, pos_did, neg_did))

            else: # Use provided negatives
                negative_docs_ids = [did for did, relevance in doc_dict.items() if relevance <= 0 and did in self.documents]
                if not negative_docs_ids: # If only positives provided, cannot create triplets
                    continue
                # Create triplets
                for pos_did in positive_docs_ids:
                    for neg_did in negative_docs_ids:
                         self.triplets.append((qid, pos_did, neg_did))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        qid, pos_did, neg_did = self.triplets[idx]

        query = self.queries[qid] # Get query text
        pos_doc = self.documents[pos_did] # Get doc text
        neg_doc = self.documents[neg_did] # Get doc text


        # Extract text from document dictionaries if needed
        if isinstance(pos_doc, dict):
            pos_doc = pos_doc.get("text", "")
        if isinstance(neg_doc, dict):
            neg_doc = neg_doc.get("text", "")

        # Encode query
        query_encoding = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=self.max_length, # Use appropriate max length
            return_tensors="pt"
        )

        # Encode positive document
        pos_encoding = self.tokenizer(
            pos_doc,
            padding="max_length",
            truncation=True,
            max_length=self.max_length, # Use appropriate max length
            return_tensors="pt"
        )

        # Encode negative document
        neg_encoding = self.tokenizer(
            neg_doc,
            padding="max_length",
            truncation=True,
            max_length=self.max_length, # Use appropriate max length
            return_tensors="pt"
        )

        return {
            "qid": qid,
            "pos_did": pos_did,
            "neg_did": neg_did,
            "query_input_ids": query_encoding["input_ids"].squeeze(),
            "query_attention_mask": query_encoding["attention_mask"].squeeze(),
            "pos_doc_input_ids": pos_encoding["input_ids"].squeeze(),
            "pos_doc_attention_mask": pos_encoding["attention_mask"].squeeze(),
            "neg_doc_input_ids": neg_encoding["input_ids"].squeeze(),
            "neg_doc_attention_mask": neg_encoding["attention_mask"].squeeze(),
        }


class RankingDataset(Dataset):
    def __init__(self, queries, documents, qrels, tokenizer, max_length=512):
        self.pairs = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.queries = queries
        self.documents = documents # Store documents ref

        # For each query, find positive documents and sample negatives
        for qid, doc_dict in tqdm(qrels.items(), desc="Building Q-D Pairs"):
            if qid not in queries:
                continue

            query = queries[qid]
            positive_docs_ids = [did for did, relevance in doc_dict.items() if relevance > 0 and did in self.documents]

            if not positive_docs_ids: # Skip queries with no relevant docs in the corpus
                continue

            all_doc_ids = list(self.documents.keys())
            candidates = random.sample(all_doc_ids, (100-len(positive_docs_ids)))

            for pos_id in positive_docs_ids:
                candidates.append(pos_id)

            for candidate_id in candidates:
                self.pairs.append((qid, candidate_id))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        qid, doc_id = self.pairs[idx]

        query = self.queries[qid] # Get query text
        doc = self.documents[doc_id] # Get doc text

        # Extract text from document dictionaries if needed
        if isinstance(doc, dict):
            doc = doc.get("text", "")

        # Encode query
        query_encoding = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=self.max_length, # Use appropriate max length
            return_tensors="pt"
        )

        # Encode document
        doc_encoding = self.tokenizer(
            doc,
            padding="max_length",
            truncation=True,
            max_length=self.max_length, # Use appropriate max length
            return_tensors="pt"
        )

        return {
            "qid": qid,
            "doc_id": doc_id,
            "query_input_ids": query_encoding["input_ids"].squeeze(),
            "query_attention_mask": query_encoding["attention_mask"].squeeze(),
            "doc_input_ids": doc_encoding["input_ids"].squeeze(),
            "doc_attention_mask": doc_encoding["attention_mask"].squeeze(),
        }



@torch.no_grad()
def encode_corpus(
    documents: Dict[str, Union[str, Dict]],
    doc_encoder: PreTrainedModel, # The document encoder part of your model
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 128,
    max_length: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Encodes all documents in the corpus."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Encoding corpus using device: {device}")

    doc_encoder.to(device)
    doc_encoder.eval()

    doc_ids = list(documents.keys())
    doc_texts = []
    # (Same text extraction logic as in the previous encode_documents)
    for doc_id in doc_ids:
        # ... extract text ...
        doc_content = documents[doc_id]
        if isinstance(doc_content, dict):
            doc_texts.append(doc_content.get("text", ""))
        elif isinstance(doc_content, str):
            doc_texts.append(doc_content)
        else:
            doc_texts.append("")


    num_docs = len(doc_ids)
    corpus_embeddings = {}

    with torch.no_grad():
        for i in tqdm(range(0, num_docs, batch_size), desc="Encoding Corpus"):
            batch_doc_ids = doc_ids[i : i + batch_size]
            batch_texts = doc_texts[i : i + batch_size]

            inputs = tokenizer(
                batch_texts, padding="max_length", truncation=True,
                max_length=max_length, return_tensors="pt"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            batch_embeddings = doc_encoder.get_embedding(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            batch_embeddings_cpu = batch_embeddings.detach().cpu()
            for doc_id, embedding in zip(batch_doc_ids, batch_embeddings_cpu):
                corpus_embeddings[doc_id] = embedding

    logger.info(f"Finished encoding corpus. {len(corpus_embeddings)} documents encoded.")
    return corpus_embeddings

@torch.no_grad() # Disable gradient calculations for inference
def encode_query(
    query_text: str,
    model: torch.nn.Module, # Expects an nn.Module with a get_embedding method
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Encodes a single query text into an embedding using the model's get_embedding method.

    Args:
        query_text: The text of the query string.
        model: An instance of the model (e.g., TripletRankerModel) which MUST have
               a method `get_embedding(self, input_ids, attention_mask)`
               that returns the desired embedding tensor.
        tokenizer: The tokenizer associated with the model.
        max_length: The maximum sequence length for tokenization.
        device: The PyTorch device ('cuda', 'cpu', etc.). Auto-detects if None.

    Returns:
        A torch.Tensor containing the query embedding, typically shape [1, hidden_size],
        residing on the specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()     # Set the model to evaluation mode
    model.to(device) # Ensure the model is on the correct device

    # Tokenize the input query text
    inputs = tokenizer(
        query_text,
        padding="max_length", # Pad to max_length
        truncation=True,      # Truncate if longer than max_length
        max_length=max_length,
        return_tensors="pt"   # Return PyTorch tensors
    )

    # Move the tokenized inputs (input_ids, attention_mask) to the target device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Pass only the required arguments explicitly to the get_embedding method
    # This avoids TypeErrors if the tokenizer adds extra keys like 'token_type_ids'
    query_embedding = model.get_embedding(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )

    # The output embedding is kept on the device it was computed on.
    # Shape is typically [batch_size=1, embedding_dimension]
    return query_embedding