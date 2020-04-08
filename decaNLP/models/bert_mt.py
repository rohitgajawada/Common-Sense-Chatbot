import os
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from .common import Embedding
from transformers import *

class BertMT(nn.Module):
    
    def __init__(self, field, args):
        super().__init__()
        self.field = field
        self.args = args
        self.pad_idx = self.field.vocab.stoi[self.field.pad_token]
        
        self.encoder_embeddings = Embedding(field, args.dimension, 
                dropout=args.dropout_ratio, project=not args.cove)
        
        self.bert_model_class = BertModel
        self.bert_tokenizer = BertTokenizer
        self.bert_pretrained_weights = 'bert-base-uncased'
        
        self.tokenizer = bert_tokenizer.from_pretrained(self.bert_pretrained_weight1s)
        self.model = bert_model_class.from_pretrained(self.bert_pretrained_weights)  
        
    def forward(self, batch):
        pass
    
        # Encode text
        input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    
        last_hidden_states = model(input_ids)[0]