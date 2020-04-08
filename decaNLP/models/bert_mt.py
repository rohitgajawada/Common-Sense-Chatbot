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
        
        self.bert_model_class = BertForQuestionAnswering
        self.bert_tokenizer = BertTokenizer
        self.bert_pretrained_weights = 'bert-base-uncased'
        
        self.tokenizer = bert_tokenizer.from_pretrained(self.bert_pretrained_weight1s)
        self.model = bert_model_class.from_pretrained(self.bert_pretrained_weights)  
        
    def forward(self, batch):
        
        context, context_lengths, context_limited, context_elmo = batch.context,  batch.context_lengths,  batch.context_limited, batch.context_elmo
        
        question, question_lengths, question_limited, question_elmo = batch.question, batch.question_lengths, batch.question_limited, batch.question_elmo
        
        answer, answer_lengths, answer_limited       = batch.answer,   batch.answer_lengths,   batch.answer_limited
        
        oov_to_limited_idx, limited_idx_to_full_idx  = batch.oov_to_limited_idx, batch.limited_idx_to_full_idx
    
        # Encode text
        context_ids = torch.tensor([self.tokenizer.encode(context, add_special_tokens=True)])  
    
        last_hidden_states = model(input_ids)[0]