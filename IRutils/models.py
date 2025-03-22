from torch import nn
from transformers import DistilBertModel


class TripletRankerModel(nn.Module):
    def __init__(self, model_name):
        super(TripletRankerModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.linear(outputs.last_hidden_state[:, 0, :])  # CLS token
        return self.sigmoid(logits)

    def get_scores(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)