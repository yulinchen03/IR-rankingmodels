import torch
import os
from torch import nn
from transformers import AutoModel

class TripletRankerModel(nn.Module):
    def __init__(self, model_name):
        super(TripletRankerModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

        # Get the hidden size from the model's configuration
        hidden_size = self.model.config.hidden_size

        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.linear(outputs.last_hidden_state[:, 0, :])  # CLS token
        return self.sigmoid(logits)

    def get_embedding(self, input_ids, attention_mask):
        """
        Returns the pooled embedding (e.g., CLS token) from the base model.
        Used for calculating distances in triplet loss or for similarity in retrieval.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Get the CLS token embedding
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return pooled_output

    def get_scores(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)


def load_model(path, model_name, device):
    model = TripletRankerModel(model_name).to(device=device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_models(model_dir, model_name, device):
    model_paths = [os.path.join(model_dir, name) for name in os.listdir(model_dir) if "full" not in name]
    models = {}

    for path in model_paths:
        print(f'Loading model {path}...')
        model = load_model(path, model_name, device)
        if 'short' in path.split('/')[-1]:
            models['short'] = model
        elif 'medium' in path.split('/')[-1]:
            models['medium'] = model
        elif 'long' in path.split('/')[-1]:
            models['long'] = model

    print(f'Ensembling models from {model_dir}!')

    return models