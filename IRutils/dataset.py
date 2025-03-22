import random
from torch.utils.data import Dataset
from tqdm import tqdm


class TripletRankingDataset(Dataset):
    def __init__(self, queries, documents, qrels, tokenizer, num_negatives, neg_sample_size=20, max_length=512):
        self.triplets = []
        self.num_negatives = num_negatives

        # For each query, find positive documents and sample negatives
        for qid, doc_dict in tqdm(qrels.items()):
            if qid not in queries:
                continue

            query = queries[qid]

            # if dataset does not contain negatives, randomly sample
            if self.num_negatives == 0:
                # Get positive documents
                positive_docs = [did for did, relevance in doc_dict.items() if did in documents]

                # Sample random documents as negatives (that aren't in positive_docs)
                all_doc_ids = list(documents.keys())
                negative_candidates = [did for did in all_doc_ids if did not in positive_docs]

                # Create triplets
                for pos_did in positive_docs:
                    # Sample a subset of negative documents
                    sampled_negatives = random.sample(negative_candidates, min(neg_sample_size, len(negative_candidates)))

                    for neg_did in sampled_negatives:
                        self.triplets.append((
                            qid,
                            pos_did,
                            neg_did,
                            query,
                            documents[pos_did],
                            documents[neg_did]
                        ))

            # if dataset already contains negatives, append them
            else:
                # Separate positive and negative documents
                positive_docs = []
                negative_docs = []

                for did, relevance in doc_dict.items():
                    if did not in documents:
                        continue

                    if relevance > 0:
                        positive_docs.append(did)
                    else:
                        negative_docs.append(did)

                # Create triplets (qid, pos_did, neg_did, query, pos_doc, neg_doc)
                for pos_did in positive_docs:
                    for neg_did in negative_docs:
                        self.triplets.append((
                            qid,
                            pos_did,
                            neg_did,
                            query,
                            documents[pos_did],
                            documents[neg_did]
                        ))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        qid, pos_did, neg_did, query, pos_doc, neg_doc = self.triplets[idx]

        # Extract text from document dictionaries
        if isinstance(pos_doc, dict):
            pos_doc = pos_doc.get("text", "")
        if isinstance(neg_doc, dict):
            neg_doc = neg_doc.get("text", "")

        # Encode anchor (query)
        anchor_encoding = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Encode query and positive document
        pos_encoding = self.tokenizer(
            query, pos_doc,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Encode query and negative document
        neg_encoding = self.tokenizer(
            query, neg_doc,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "qid": qid,
            "pos_did": pos_did,
            "neg_did": neg_did,
            "anchor_input_ids": anchor_encoding["input_ids"].squeeze(),
            "anchor_attention_mask": anchor_encoding["attention_mask"].squeeze(),
            "positive_input_ids": pos_encoding["input_ids"].squeeze(),
            "positive_attention_mask": pos_encoding["attention_mask"].squeeze(),
            "negative_input_ids": neg_encoding["input_ids"].squeeze(),
            "negative_attention_mask": neg_encoding["attention_mask"].squeeze(),
        }