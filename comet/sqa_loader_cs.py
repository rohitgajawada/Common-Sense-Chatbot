import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import jsonlines
import tqdm 
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMultipleChoice,
    BertTokenizer,
    RobertaConfig,
    RobertaForMultipleChoice,
    RobertaTokenizer,
    XLNetConfig,
    XLNetForMultipleChoice,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
import sys
import argparse
import torch

import spacy

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive

class SQA(object):
    """A single training/test example for the SQA dataset."""
    def __init__(self,
                 input_id,
                 contexts,
                 question,
                 choice_1,
                 choice_2,
                 choice_3,
                 label = None):
        self.input_id = input_id
        self.contexts = contexts
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
            f"contexts: {self.contexts}",
            f"question: {self.question}",
            f"choice_1: {self.choices[0]}",
            f"choice_2: {self.choices[1]}",
            f"choice_3: {self.choices[2]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)

class InputFeatures(object):
    def __init__(self,choices_features, label):
        # self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


# def read_sqa_examples(input_file, label_file = None, is_training = None):
#     with jsonlines.open(input_file) as reader:
#         lines = list(reader)
    
#     if(label_file != None):
#         with open(label_file) as f:
#             reader = f.read()
#             labels = list(reader)

#     examples = []
#     for i in range(len(lines)):
#         examples += [
#             SQA(
#                 context = lines[i]["context"],
#                 question = lines[i]["question"],
#                 choice_1 = lines[i]["answerA"],
#                 choice_2 = lines[i]["answerB"],
#                 choice_3 = lines[i]["answerC"],
#                 label = labels[i] if is_training else None
#             ) 
#         ]
#     return examples

class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    # def get_test_examples(self, data_dir):
    #     """Gets a collection of `InputExample`s for the test set."""
    #     raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class SQAProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        lines, labels = self._read_json("socialiqa-train-dev/train.jsonl", "socialiqa-train-dev/train-labels.lst")
        return self._create_examples(lines, labels, True)

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines, labels =self._read_json("socialiqa-train-dev/dev.jsonl","socialiqa-train-dev/dev-labels.lst")
        return self._create_examples(lines, labels, False)

    # def get_test_examples(self, data_dir):
    #     """See base class."""
    #     raise ValueError(
    #         "For swag testing, the input file does not contain a label column. It can not be tested in current code"
    #         "setting!"
    #     )
    #     return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]
        

    def _read_json(self, input_file, label_file):
        with jsonlines.open(input_file) as reader:
            lines = list(reader)

        with open(label_file) as f:
            reader = f.read().splitlines()
            labels = list(reader)
        
        return lines, labels

    def _create_examples(self, lines, labels, is_training):
        """Creates examples for the training and dev sets."""
        # if type == "train" and lines[0][-1] != "label":
        #     raise ValueError("For training, the input file must contain a label column.")
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

        def augment(article):
            category = "all"
            entity_list = nlp(article)
            input_event = article
            replaced = []
            replacement_list = ["PersonX", "PersonY"]
            r = 0
            for entity in entity_list.ents:
    
                replaced += [entity.text]
                
                input_event = input_event.replace(entity.text, replacement_list[r])
                r += 1
                if(r == 2):
                    break

            outputs = interactive.get_atomic_sequence(
                input_event, model, sampler, data_loader, text_encoder, category)
            for key in outputs:
                prefix = ""
                if(key[0] == "o"):
                    if(key == "oEffect"):
                        prefix = " Everyone else "
                    elif (key == "oReact"):
                        prefix = "They are "
                    elif(key == "oWant"):
                        prefix = "They want "
                else:
                    if(len(replaced)!= 0):
                        prefix = replaced[0]
                    else:
                        prefix = "Person"
                    if(key == "xAttr"):
                        prefix += " is "
                    elif(key == "xEffect"):
                        prefix += " "
                    elif(key == "xIntent"):
                        prefix += " intends "
                    elif(key == "xReact"):
                        prefix += " is "
                    elif(key == "xNeed"):
                        prefix += " needs "
                    elif(key == "xWant"):
                        prefix += " wants "
                    
                for j in range(5):
                    
                    if(outputs[key]["beams"][j] != 'none'):
                        comet_inf = outputs[key]["beams"][j] 
                        if(len(replaced) > 0):
                            comet_inf = comet_inf.replace("personx", replaced[0])
                            if(len(replaced) > 1):
                                comet_inf = comet_inf.replace("persony", replaced[1])

                        article += prefix + (comet_inf) + ". "
                        break
            
            return article
        

        examples = []
        # print(lines[5]["context"])
        # print("jere")
        # aug_context = augment(lines[5]["context"])
        for i in range(len(lines)):
            # print(lines[i]["context"])
            aug_context = augment(lines[i]["context"])
            examples += [
                SQA( input_id = str(i),
                    contexts = aug_context,
                    question = lines[i]["question"],
                    choice_1 = lines[i]["answerA"],
                    choice_2 = lines[i]["answerB"],
                    choice_3 = lines[i]["answerC"],
                    label = str(int(labels[i])-1) 
                ) 
            ]
        return examples



def convert_examples_to_features(
    examples,
    label_list,
    max_length,
    tokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) :
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        # print(example)
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for choice_idx, (context, choices) in enumerate(zip(example.contexts, example.choices)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", choices)
            else:
                text_b = example.question + " " + choices

            inputs = tokenizer.encode_plus(
                text_a, text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True
            )
            # if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            #     print(
            #         "Attention! you are cropping tokens (swag task is ok). "
            #         "If you are training ARC and RACE and you are poping question + options,"
            #         "you need to try to use a bigger max seq length!"
            #     )

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))
        label = label_map[example.label]

        # if ex_index < 2:
        #     print("*** Example ***")
        #     print("race_id: {}".format(example.example_id))
        #     for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
        #         print("choice: {}".format(choice_idx))
        #         print("input_ids: {}".format(" ".join(map(str, input_ids))))
        #         print("attention_mask: {}".format(" ".join(map(str, attention_mask))))
        #         print("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
        #         print("label: {}".format(label))

        features.append(InputFeatures(choices_features=choices_features, label=label,))

    return features


def main():
    max_seq_length = 50
    processor = SQAProcessor()
    config_class, model_class, tokenizer_class = (BertConfig, BertForMultipleChoice, BertTokenizer)

    tokenizer = tokenizer_class.from_pretrained(
        "bert-base-uncased" ,
        do_lower_case=False,
        cache_dir= None,
    )

    examples = processor.get_train_examples('socialiqa-train-dev/train.jsonl')
    label_list = processor.get_labels()
    features = convert_examples_to_features(
            examples,
            label_list,
            max_seq_length,
            tokenizer,
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
        )
    

if __name__ == "__main__":
    main()
