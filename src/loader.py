"""
Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import pathlib
import argparse
import shlex
import deepcopy
from collections import defaultdict, Counter, OrderedDict, namedtuple
from typing import List, Tuple, Iterable, Union

import numpy as np
import pandas as pd
from ordered_set import OrderedSet
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import transformers

from logging_util import ColoredLog
from vocab import Vocabulary

plt.rcParams.update({"figure.figsize": (12, 8), "font.size": 20})
project_root = pathlib.Path(__file__).parents[1]


# Prediction for each prefix contains token level intents, sentence level intents, and slots
Prediction = namedtuple("Prediction", ["sent_intent", "token_intent", "slots"])


class IncrementalQuery(object):

    """An object to represent incrementalized queries."""

    def __init__(self, sent_id: str, tokens: List[str], intents: List[str], slots: List[str]):
        """Initialize an incrementalized query instance.

        :sent_id: sentence id
        :tokens: query tokens
        :intents: golden intents
        :slots: golden slots

        """
        self.sent_id = sent_id
        self.tokens = tokens
        self.intents = intents
        self.slots = slots
        self.predictions = {}
        self.logger = ColoredLog(__name__)

    @property
    def id(self):
        return self.sent_id

    @property
    def query(self):
        return " ".join(self.tokens[1:-1])

    @property
    def size(self):
        return len(self.tokens) - 2

    def update_predictions(self, sent_intent: List[str], token_intent: List[str], slots: List[str]) -> None:
        """ Update predictions from model.

        :sent_intent: sentence level intent
        :token_intent: token level intent
        :slots: slots prediction

        """

        num_item = len(slots)
        assert num_item <= len(self.tokens)
        assert num_item != len(self.tokens) - 1, "Full query without <EOS> tag, should not exist."

        prefix_size = num_item - 2 if num_item == len(self.tokens) else num_item - 1
        self.predictions[prefix_size] = Prediction._make([sent_intent, token_intent, slots])

    def get_prediction(self, prefix_size: Union[int, None]=None) -> Union[Tuple, None]:
        """Get the prediction results for the query.

        :prefix_size: a particular prefix length; if None, use full query

        """
        try:
            if prefix_size is None:
                prefix_size = self.size
            return tuple(getattr(self.predictions[prefix_size], f) for f in Prediction._fields)
        except Exception:
            prefix = " ".join(self.tokens[1:prefix_size+1])
            self.logger.error(f"Predictions for prefix '{prefix}' not set yet.")

    def info(self, prefix_size=None) -> None:
        """ Print information for particular prefix_size. """

        if prefix_size is None:
            prefix_size = self.size
        num_item = prefix_size + 1 if prefix_size < self.size else prefix_size + 2

        assert 0 < prefix_size <= self.size, "prefix size larger than max length"

        tokens = self.tokens[:num_item]
        intent = ["#".join(self.intents)] * num_item
        slots = self.slots[:num_item]

        if self.get_prediction(prefix_size):
            pred_i, pred_t, pred_s = self.get_prediction(prefix_size)
            pred_i = ["#".join(pred_i)] * num_item
        else:
            pred_i, pred_t, pred_s = [["NA"] * num_item] * 3

        print_data = [tokens, intent, slots, pred_i, pred_t, pred_s]
        caption = f"sent id: {self.id}\nsub_id: {prefix_size}\nquery: {' '.join(self.tokens[1:prefix_size+1])}"
        header = ["token", "gold_int", "gold_slot", "pred_int", "pred_tkl", "pred_slot"]
        self.logger.info(print_data, caption=caption, transpose=True, header=header)


class Padding(object):
    """ An transformation object: add padding. """

    def __init__(self, pad_token="[PAD]", max_length=20):
        """ Initialize the transformation.

        :pad_token: specify token representation to be used as padding
        :max_length: length after padding
        """
        self._pad_token = pad_token
        self._max_length = max_length

    def __call__(self, sample):
        padded_tokens = sample.tokens + [self._pad_token] * (self._max_length - len(sample.tokens))
        return {"padded_tokens": padded_tokens, "intent": sample.intents}


class ToTensor(object):
    """ An transformation object: encode queries to numerial tensors. """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __call__(self, sample):
        encoded = self._tokenizer.encode(sample["padded_tokens"])
        return {"tokens": encoded, "intent": sample["intent"]}


class IncrementalDataset(Dataset):

    """Incremental Torch Dataset object. Allow filtering with different initial prefix size, prefix incremental size, or full utterance only."""

    def __init__(self, data, incremental_size=1, initial_prefix=1, transform=None):
        """Create an incremental dataset object.

        :data: an incremental dataframe object
        :incremental_size: how many additional tokens to add at each incremental step
        :initial_prefix: how many tokens to pass to the model at the initial step
        :transform: transformation performed to each point in the dataset

        """
        super(IncrementalDataset, self).__init__()

        self._data = data
        self._incremental_size = incremental_size
        self._initial_prefix = initial_prefix
        self._transform = transform
        self.sampler = self.incremental_filtering()

    def incremental_filtering(self):
        """ Create sub sampler if not all incrementalized instances are included. """

        sub_sample = [True] * len(self._data)
        for j, entry in enumerate(self._data):
            sub_id = entry.sub_id
            if sub_id == len(entry.tokens) - 2:
                continue
            elif int(sub_id) < self._initial_prefix or (int(sub_id) - self._initial_prefix) % self._incremental_size != 0:
                sub_sample[j] = False

        return np.arange(len(sub_sample))[sub_sample]

    def __len__(self):
       return len(self._data)

    def __getitem__(self, idx):
        if self._transform is not None:
            return self._transform(self._data[idx].tokens), self._transform(self._data[idx].slots), self._data[idx].intents, f"{self._data[idx].sent_id}-{self._data[idx].sub_id}", self._data[idx].anticipation, self._data[idx].entropy
        else:
            return self._data[idx].token_encode, self._data[idx].slot_encode, self._data[idx].intent_encode, f"{self._data[idx].sent_id}-{self._data[idx].sub_id}", self._data[idx].anticipation, self._data[idx].entropy


class LMDist(object):

    """Stores and returns the empirical distribution of anticipated tokens for any prefix (needed for training only). """

    def __init__(self, ngram: int=3):
        self.ngram = ngram
        self.prefix_dist = defaultdict(lambda: defaultdict(int))

    def update(self, tokens):
        """ Update the empirical distribution record for subsequent tokens given a fixed-size prefix.

        :tokens: specify the prefix
        """
        if len(tokens) > self.ngram:
            self.prefix_dist[" ".join(tokens[-self.ngram-1:-1])][tokens[-1]] += 1
        elif len(tokens) > 1:
            self.prefix_dist[tokens[:-1]][tokens[-1]] += 1

    def getDist(self, input_tokens, vocab):
        """ Returns the empirical distribution.

        :input_tokens: specify which empirical distributions to look for after receiving these tokens
        :vocab: a Vocabulary instance that can encode tokens
        """
        counter = self.prefix_dist[" ".join(input_tokens)]
        total = sum(counter.values())
        dist_vec = np.zeros(len(vocab))
        keys = list(counter.keys())
        index = vocab.encode(keys)
        for k, i in zip(keys, index):
            dist_vec[i] = float(counter[k]) / total
        return dist_vec

    def save(self, filepath):
        torch.save(self.prefix_dist, filepath)


class IncrementalDataLoader(object):

    """Read in the preprocessed data formats with incremental data entries."""

    def __init__(self, data_name, file, rebuild_vocab:bool =True, vocab_dir: str="vocab", anticipation_size: int=2, verbose: int=3):
        """Read in the data file and convert it to data frame format.

        :data_name: one of the four target dataset names, atis, top, fbml, or snips
        :file: one of the three dataset types, train, test, or dev
        :vocab_dir: the save/load vocab directory
        :rebuild_vocab: whether to rebuild the vocabulary
        :anticipation_size: number of tokens to anticipate
        :verbose level: 0->critical, 1->error, 2->warning, 3->info, 4+->debug

        """

        self.data_root = os.path.join(project_root, "data", data_name)
        self.data_name = data_name
        self.file = file
        self.vocab_dir = os.path.join(self.data_root, vocab_dir)
        self.rebuild_vocab = rebuild_vocab
        self.anticipation_size = anticipation_size
        self.logger = ColoredLog(__name__, verbose=verbose)

        self.prefix_set = defaultdict(list) # key is prefix size, value is a list of prefix with `key` size
        self.prefix_intent = defaultdict(list) # key is prefix, value is the associated intent with this `key`
        self.all_intent = defaultdict(int) # key is intent, value is the count for this `key`

        if self.rebuild_vocab:
            special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]","[UNK]",  "<BOS>", "<EOS>"]
            self.token_vocab = Vocabulary("token", special_tokens)
            self.intent_vocab = Vocabulary("intent")
            self.slot_vocab = Vocabulary("slot")
            self.lmdist = LMDist()
        else:
            self.token_vocab = torch.load(os.path.join(self.vocab_dir, "token.vocab"))
            self.intent_vocab = torch.load(os.path.join(self.vocab_dir, "intent.vocab"))
            self.slot_vocab = torch.load(os.path.join(self.vocab_dir, "slot.vocab"))
            self.lmdist = torch.load(os.path.join(self.vocab_dir, "lmdist.pkl"))

        self.read_file()
        self.calculate_prefix_intent_dist()

    def read_file(self) -> None:
        """ Read in data file. """

        self.data_points = dict()
        self.max_length = 0
        with open(os.path.join(self.data_root, self.file), "r") as f:
            get_id = True
            sep_len = len(self.file[:-4]) + 6
            tokens = []
            slots = []
            for lines in f:
                l = lines.strip().split()
                if len(l) == 0: # instance separation
                    get_id = True
                    tokens = []
                    slots = []
                elif len(l) == 1: # either sentence id or intent
                    if get_id: # sentence id
                        sent_id = l[0][:sep_len]
                        current_size = l[0][sep_len+1:]
                        get_id = False
                    else: # intent
                        intent_list = l[0].split("#")
                        if current_size == "full": # updates for full quries only
                            self.data_points[sent_id] = IncrementalQuery(sent_id, tokens, intent_list, slots)
                            for s in intent_list:
                                self.all_intent[s] += 1
                            if self.rebuild_vocab:
                                self.token_vocab.add_instance(tokens)
                                self.intent_vocab.add_instance(intent_list)
                                self.slot_vocab.add_instance(slots)
                            if len(tokens) > self.max_length:
                                self.max_length = len(tokens)
                        else: # update prefix related values
                            if self.rebuild_vocab:
                                self.lmdist.update(tokens[1:])
                            prefix = " ".join(tokens[1:])
                            current_size = int(current_size)
                            self.prefix_set[current_size].append(prefix)
                            for s in intent_list:
                                self.prefix_intent[prefix].append(s)
                else: # token/slot pair
                    tokens.append(l[0])
                    slots.append(l[1])

        if self.rebuild_vocab:
            if not os.path.exists(self.vocab_dir):
                os.mkdir(self.vocab_dir)

            self.token_vocab.save(self.vocab_dir)
            self.intent_vocab.save(self.vocab_dir)
            self.slot_vocab.save(self.vocab_dir)
            self.lmdist.save(os.path.join(self.vocab_dir, "lmdist.pkl"))

        self.logger.info(f"max length: {self.max_length}")
        self.logger.info(f"vocab size: {len(self.token_vocab)}")
        self.logger.info(f"intent size: {len(self.intent_vocab)}")
        self.logger.info(f"slot size: {len(self.slot_vocab)}")
        self.vocab_dict = {"token": self.token_vocab, "intent": self.intent_vocab, "slot": self.slot_vocab}

    def calculate_prefix_intent_dist(self) -> None:
        """ Calculate intent distribution for each prefix. """

        self.freq_dict = {}
        self.freq_dict_norm = {}
        for p, v in self.prefix_intent.items():
            counter = Counter(v)
            self.freq_dict[p] = {k: float(d)/sum(counter.values()) for k, d in counter.items()}
            normed = {k: float(d)/self.all_intent[k] for k, d in counter.items()}
            self.freq_dict_norm[p] = {k: d/sum(normed.values()) for k, d in normed.items()}

    @property
    def dataset(self) -> List:
        try:
            return self.data_entries
        except Exception:
            return self.construct_dataset(self.anticipation_size)

    def construct_dataset(self, anticipation_size: int) -> List:
        """ Consturct an object for torch Dataset.

        :anticipation_size: how many tokens to anticipate

        """

        DataEntry = namedtuple("DataEntry", ["sent_id", "sub_id", "tokens", "intents", "slots", "token_encode", "intent_encode", "slot_encode", "anticipation", "entropy"])
        self.data_entries = []
        for sent, query in self.data_points.items():
            for i in list(range(1, query.size)) + [query.size + 1]:
                if i == query.size + 1:
                    sub_id = i - 1
                else:
                    sub_id = i

                tok_enc = self.token_vocab.encode(query.tokens[:i+1])
                int_enc = self.intent_vocab.encode(query.intents)
                slt_enc = self.slot_vocab.encode(query.slots[:i+1])
                entry = [sent, sub_id, query.tokens[:i+1], query.intents, query.slots[:i+1]]
                entry.extend([tok_enc, int_enc, slt_enc])

                # adding anticipation tokens
                anticipation_padded = query.tokens + ["[PAD]"] * anticipation_size
                anticipation_tokens = anticipation_padded[i+1 : i+1+anticipation_size]
                entry.append(anticipation_tokens)
                # adding entropy values
                # very large number for special tokens <BOS> and <EOS>
                entropy_vec = [self.entropy(" ".join(query.tokens[1:j+1])) for j in range(1, i+1)]
                if i == query.size + 1:
                    entropy_vec[-1] = 999
                entry.append([999] + entropy_vec)
                self.data_entries.append(DataEntry._make(entry))

        return self.data_entries

    def entropy(self, prefix: str, normalize: bool=True) -> float:
        """ Calculate static prefix entropy.

        :prefix: prefix of interest
        :normalize: whether to normalize by the population intent distribution

        """

        if prefix in self.prefix_intent:
            if normalize:
                return np.sum([-x*np.log2(x) for x in self.freq_dict_norm[prefix].values()])
            else:
                return np.sum([-x*np.log2(x) for x in self.freq_dict[prefix].values()])
        else:
            # max entropy for OOV tokens
            n = len(self.intent_vocab)
            return -(1./n)*np.log2(1./n) * n

    def batch_delivery(self, batch_size: int=16):
        """ Create DataLoader batch to be fed into model. """
        self.ids = IncrementalDataset(self.dataset)
        return DataLoader(self.ids, batch_size=batch_size, sampler=SubsetRandomSampler(self.ids.sampler), shuffle=False, collate_fn=self.__collate_fn)

    @staticmethod
    def add_padding(texts, items=None, numeric=True):
        """ Sort `texts` from longest to shortest and .

        :texts: tokenized queries to be sorted according to length
        :items: other items that need to be sorted according to the order of `texts`
        :numeric: whether the arguments are passed in in string or numeric form
        """

        len_list = [len(text) for text in texts]
        max_len = max(len_list)

        # Get sorted index of len_list.
        sorted_index = np.argsort(len_list)[::-1]

        trans_texts, seq_lens, trans_items = [], [], None
        if items is not None:
            trans_items = [[] for _ in range(0, len(items))]

        for index in sorted_index:
            seq_lens.append(deepcopy(len_list[index]))
            trans_texts.append(deepcopy(texts[index]))
            if numeric:
                trans_texts[-1].extend([0] * (max_len - len_list[index]))
            else:
                trans_texts[-1].extend(['<PAD>'] * (max_len - len_list[index]))

            if items is not None:
                for item, (o_item, required) in zip(trans_items, items):
                    item.append(deepcopy(o_item[index]))
                    if required:
                        if numeric:
                            item[-1].extend([0] * (max_len - len_list[index]))
                        else:
                            item[-1].extend(['<PAD>'] * (max_len - len_list[index]))

        if items is not None:
            return trans_texts, trans_items, seq_lens, sorted_index
        else:
            return trans_texts, seq_lens

    @staticmethod
    def __collate_fn(batch):
        """ A helper function to arrange batch in DataLoader. """

        n_entity = len(batch[0])
        modified_batch = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(batch)):
            for jdx in range(0, n_entity):
                modified_batch[jdx].append(batch[idx][jdx])

        return modified_batch
