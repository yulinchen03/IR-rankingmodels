# --- START OF FILE dataset.py ---

import random
import time
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



# Check PyTorch version for torch.compile
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
use_torch_compile = (TORCH_MAJOR >= 2)

@torch.no_grad() # Keep no_grad for inference efficiency
def encode_corpus(
    documents: Dict[str, Union[str, Dict]],
    doc_encoder: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 1024,  # <--- INCREASED default, tune this!
    max_length: int = 512,
    device: Optional[torch.device] = None,
    use_amp: bool = True,      # <--- Added flag for AMP
    compile_model: bool = True, # <--- Added flag for torch.compile
) -> Dict[str, torch.Tensor]:
    """
    Encodes all documents in the corpus with optimizations.

    Args:
        documents: Dictionary mapping doc_id to document text or dict containing text.
        doc_encoder: The document encoder model.
        tokenizer: The tokenizer.
        batch_size: Number of documents to process at once. Tune for your GPU RAM.
        max_length: Max sequence length for tokenizer.
        device: Target device (e.g., torch.device("cuda")). Auto-detects if None.
        use_amp: Whether to use Automatic Mixed Precision (AMP) for speed/memory savings.
        compile_model: Whether to use torch.compile (PyTorch 2.0+) for potential speedup.

    Returns:
        Dictionary mapping doc_id to its embedding tensor on CPU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Encoding corpus using device: {device}")
    if use_amp and device.type != 'cuda':
        logger.warning("AMP is requested but device is not CUDA. Disabling AMP.")
        use_amp = False
    if compile_model and not use_torch_compile:
        logger.warning("torch.compile requested but PyTorch version < 2.0. Disabling compile.")
        compile_model = False
    if compile_model and device.type == 'cpu':
        logger.warning("torch.compile on CPU might be slow or have limited backend support. Proceeding...")


    # --- Model Preparation ---
    doc_encoder.to(device)
    doc_encoder.eval()

    # Apply torch.compile if requested and possible
    if compile_model:
        logger.info("Applying torch.compile to the document encoder...")
        try:
            # Common modes:
            # 'default': Good balance
            # 'reduce-overhead': Faster compilation, potentially less speedup
            # 'max-autotune': Slower compilation, potentially more speedup
            start_compile_time = time.time()
            doc_encoder = torch.compile(doc_encoder, mode='default')
            logger.info(f"torch.compile applied successfully (took {time.time() - start_compile_time:.2f}s). Note: First batch will include warmup/compilation time.")
        except Exception as e:
            logger.error(f"torch.compile failed: {e}. Proceeding without compilation.")
            compile_model = False # Fallback

    # --- Data Preparation ---
    logger.info("Preparing document texts...")
    doc_ids = list(documents.keys())
    doc_texts: List[str] = [] # Explicitly type hint
    for doc_id in doc_ids:
        doc_content = documents[doc_id]
        text_to_append = "" # Default to empty string
        if isinstance(doc_content, dict):
            text_to_append = doc_content.get("text", "") # Safely get text
        elif isinstance(doc_content, str):
            text_to_append = doc_content
        # Handle potential None or other unexpected types gracefully if necessary
        if text_to_append is None:
            text_to_append = ""
        doc_texts.append(text_to_append)

    num_docs = len(doc_ids)
    corpus_embeddings: Dict[str, torch.Tensor] = {} # Explicit type hint

    logger.info(f"Starting encoding of {num_docs} documents with batch size {batch_size}...")
    # Check tokenizer type
    if not getattr(tokenizer, 'is_fast', False): # Use getattr for safety
         logger.warning("Tokenizer is not a 'Fast' tokenizer. Consider installing 'tokenizers' and ensuring a Fast tokenizer is loaded for better performance.")

    # --- Encoding Loop ---
    for i in tqdm(range(0, num_docs, batch_size), desc="Encoding Corpus", leave=False):
        batch_doc_ids = doc_ids[i : i + batch_size]
        batch_texts = doc_texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding="longest", # <--- Use 'longest' or 'max_length'
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            # return_attention_mask=True # Ensure mask is returned, usually default with pt
        )

        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Use AMP context manager if enabled
        with torch.amp.autocast('cuda', enabled=use_amp):
            # Assuming your model has a method 'get_embedding' or similar
            # If not, use the standard forward pass and extract embeddings
            # e.g., outputs = doc_encoder(**inputs)
            #      batch_embeddings = outputs.last_hidden_state[:, 0] # CLS token example
            try:
                 batch_embeddings = doc_encoder.get_embedding(
                     input_ids=inputs['input_ids'],
                     attention_mask=inputs['attention_mask']
                 )
            except AttributeError:
                 # Fallback to standard model call if get_embedding doesn't exist
                 outputs = doc_encoder(**inputs)
                 # Adapt this extraction based on your model and desired embedding type
                 # Example: CLS token pooling
                 batch_embeddings = outputs.last_hidden_state[:, 0]
                 # Example: Mean pooling (needs attention mask)
                 # token_embeddings = outputs.last_hidden_state
                 # input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                 # sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                 # sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                 # batch_embeddings = sum_embeddings / sum_mask


        # Move embeddings to CPU right away to free GPU memory for next batch
        batch_embeddings_cpu = batch_embeddings.detach().cpu()
        for doc_id, embedding in zip(batch_doc_ids, batch_embeddings_cpu):
            corpus_embeddings[doc_id] = embedding

    logger.info(f"Finished encoding corpus. {len(corpus_embeddings)} documents encoded.")
    return corpus_embeddings
