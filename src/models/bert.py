import torch.nn as nn
from transformers import BertForSequenceClassification
from src.layers import AnalogLinear
class AnalogBertClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=2, physics_engine=None):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.physics_engine = physics_engine
        self._convert_to_analog(self.bert)
    def _convert_to_analog(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                new_layer = AnalogLinear(child.in_features, child.out_features, bias=(child.bias is not None), physics_engine=self.physics_engine)
                new_layer.weight.data = child.weight.data
                if child.bias is not None: new_layer.bias.data = child.bias.data
                setattr(module, name, new_layer)
            else: self._convert_to_analog(child)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
