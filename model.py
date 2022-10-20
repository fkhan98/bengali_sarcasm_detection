import torch

from torch import nn
from transformers import BertModel
# from transformers.modeling_utils import PreTrainedModel
# from transformers.modeling_outputs import SequenceClassifierOutput
# from transformers import BertPreTrainedModel

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('sagorsarker/bangla-bert-base')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim = 1) 
        
    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.softmax(linear_output)
        
        return final_layer