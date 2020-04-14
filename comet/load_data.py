import tensorflow_datasets as tfds

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


import os
import sys
import argparse
import torch

import spacy

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive

import tensorflow as tf



tfds_examples = tfds.load("squad")
print(tfds_examples)

device = "cpu"
comet_model = "pretrained_models/atomic_pretrained_model.pickle"
sampling_algo = "beam-5"


opt, state_dict = interactive.load_model_file(comet_model)

data_loader, text_encoder = interactive.load_data("atomic", opt)

n_ctx = data_loader.max_event + data_loader.max_effect
n_vocab = len(text_encoder.encoder) + n_ctx
model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)
nlp = spacy.load("en_core_web_sm")

if device != "cpu":
    cfg.device = int(device)
    cfg.do_gpu = True
    torch.cuda.set_device(cfg.device)
    model.cuda(cfg.device)
else:
    cfg.device = "cpu"

sampling_algorithm = sampling_algo

sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)
new_list = {}

def augment(article):
    title_new = (article.numpy().decode('UTF-8'))
    category = "oEffect"
    entity_list = nlp(title_new)
    input_event = title_new
    replacement_list = ["PersonX", "PersonY", "PersonZ"]
    r = 0
    for entity in entity_list.ents:
        if entity.label_ == 'PERSON' or entity.label_ == 'NORP':
            input_event = input_event.replace(entity.text, replacement_list[r])
            r += 1
            if(r == 3):
                break

    outputs = interactive.get_atomic_sequence(
        input_event, model, sampler, data_loader, text_encoder, category)

    for key in outputs:
        article = title_new + ((outputs[key]["beams"][0]))
    return article
    

def process_example(example):

    example['context'] = tf.py_function(func=augment,
                                    inp=[example['context']],
                                    Tout=tf.string)
    return example
 

tfds_examples["train"] = tfds_examples["train"].map(lambda x: process_example(x))




examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=False)

# features = squad_convert_examples_to_features(
#     examples=examples,
#     tokenizer=tokenizer,
#     max_seq_length=max_seq_length,
#     doc_stride=args.doc_stride,
#     max_query_length=max_query_length,
#     is_training=not evaluate,
# )
