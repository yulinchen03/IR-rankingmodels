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

    def get_scores(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)