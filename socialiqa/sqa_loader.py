import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import jsonlines

class SQA(object):
    """A single training/test example for the SQA dataset."""
    def __init__(self,
                 context,
                 question,
                 choice_1,
                 choice_2,
                 choice_3,
                 label = None):
        self.context = context
        self.question = question
        self.choices = [
            choice_1,
            choice_2,
            choice_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"context: {self.context}",
            f"question: {self.question}",
            f"choice_1: {self.choices[0]}",
            f"choice_2: {self.choices[1]}",
            f"choice_3: {self.choices[2]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)

def read_sqa_examples(input_file, label_file, is_training):
    with jsonlines.open(input_file) as reader:
        lines = list(reader)
    
    with open(label_file) as f:
        reader = f.read()
        labels = list(reader)


    # if is_training and lines[0][-1] != 'label':
    #     raise ValueError(
    #         "For training, the input file must contain a label column."
    #     )
    examples = []
    for i in range(len(lines)):
        examples += [
            SQA(
                context = lines[i]["context"],
                question = lines[i]["question"],
                choice_1 = lines[i]["answerA"],
                choice_2 = lines[i]["answerB"],
                choice_3 = lines[i]["answerC"],
                label = labels[i] if is_training else None
            ) 
        ]
    return examples


def main():
    
    train_examples = read_sqa_examples('socialiqa-train-dev/train.jsonl', 'socialiqa-train-dev/train-labels.lst', is_training = True)
    

if __name__ == "__main__":
    main()
