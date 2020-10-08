import os
import sys
import pathlib
import argparse
import shlex
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (12, 8), "font.size": 20})

import torch
import transformers
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from ordered_set import OrderedSet

try:
    print(__file__)
except NameError:
    __file__ = "/home/agneshu_google_com/nlu-isp/src/loader.py"

source_root = pathlib.Path(__file__).parent
sys.path.append(source_root)
from logging_util import ColoredLog

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_log", type=bool, default=False)
parser.add_argument("--cpu", action="store_true", default=False)
# Training parameters.
parser.add_argument("--data", "-d", type=str, default="atis")
parser.add_argument("--file", "-f", type=str, default="train.txt")
parser.add_argument("--verbose", "-v", action="count", default=0)
parser.add_argument("--save_dir", "-s", type=str, default="save")
parser.add_argument("--init_prefix", "-ip", type=int, default=1)
parser.add_argument("--incre_size", "-is", type=int, default=1)

arg_string = "-vvvv"
args = parser.parse_args(shlex.split(arg_string))

if args.wandb_log:
    import wandb
    wandb.init(project="test", config=args)
    config = wandb.config
else:
    config = args

VERBOSE = config.verbose
logger = ColoredLog(__file__, verbose=VERBOSE)

class IncrementalDataReader(object):

    """Read in the preprocessed data formats with incremental data entries."""

    def __init__(self, data_name, file, full_utt_only=False):
        """Read in the data file and convert it to data frame format.

        :data_name: one of the four target dataset names, atis, tops, fbml, or snips
        :file: one of the three dataset types, train, test, or dev
        :full_utt_only: whether to filter out partial utterances

        """
        data_root = os.path.join(pathlib.Path(__file__).parents[1], "data", data_name)
        self._data_name = data_name
        self._file = file
        self._is_train = self._file.startswith("train")
        self._full_utt_only = full_utt_only

        self.prefix_set = defaultdict(lambda: defaultdict(list))
        self.prefix_intent = defaultdict(lambda: defaultdict(list))
        self.all_intent = defaultdict(int)

        self.token_vocab = OrderedSet(("[CLS]", "[SEP]", "[PAD]", "[MASK]","[UNK]",  "<BOS>", "<EOS>"))
        self.intent_vocab = OrderedSet()
        self.slot_vocab = OrderedSet()

        self.logger = ColoredLog(__name__, verbose=VERBOSE)
        self.max_length = 0

        self.data_points = defaultdict(list)
        with open(os.path.join(data_root, self._file), "r") as f:
            get_id = True
            tokens = []
            tags = []
            for lines in f:
                l = lines.strip().split()
                if len(l) == 0:
                    get_id = True
                    tokens = []
                    tags = []
                elif len(l) == 1:
                    if get_id:
                        self.data_points["id"].append(l[0])
                        #  current_id = l[0][:11]
                        current_size = l[0][12:]
                        get_id = False
                    else:
                        intent_list = l[0].split("#")
                        self.data_points["tokens"].append(tokens)
                        self.data_points["tags"].append(tags)
                        self.data_points["partial"].append(" ".join(tokens))
                        self.data_points["intents"].append(intent_list)
                        if not current_size.startswith("full"):
                            self.prefix_intent[f"size_{current_size}"][" ".join(tokens[1:])].append(l[0])
                        else:
                            for i in intent_list:
                                self.all_intent[i] += 1
                                self.intent_vocab.add(i)
                            if len(tokens) > self.max_length:
                                self.max_length = len(tokens)
                else:
                    if l[0] not in self.token_vocab and self._is_train:
                        self.token_vocab.add(l[0])
                    if l[1] not in self.slot_vocab and self._is_train:
                        self.slot_vocab.add(l[1])
                    if not current_size.startswith("full"):
                        self.prefix_set[f"size_{int(current_size) - 1}"][" ".join(tokens[1:])].append(l[0])
                    tokens.append(l[0])
                    tags.append(l[1])

        #  self.vocab_size = len(self.vocab)
        self.logger.info(f"max length: {self.max_length}")

        #  if self._is_train and not os.path.exists(self.vocab_path):
        if self._is_train:
            self.vocab_path = os.path.join(data_root, "token.vocab")
            self.intent_path = os.path.join(data_root, "intent.vocab")
            self.slot_path = os.path.join(data_root, "slot.vocab")
            self.logger.info(f"vocab size: {len(self.token_vocab)}")
            self.logger.info(f"intent size: {len(self.intent_vocab)}")
            self.logger.info(f"slot size: {len(self.slot_vocab)}")
            with open(self.vocab_path, "w") as f:
                f.write("\n".join(self.token_vocab))
            with open(self.intent_path, "w") as f:
                f.write("\n".join(self.intent_vocab))
            with open(self.slot_path, "w") as f:
                f.write("\n".join(self.slot_vocab))

        self.dataset = pd.DataFrame(data=self.data_points, index=self.data_points["id"])
        if self._full_utt_only:
            self.dataset = self.dataset.filter(regex="full$", axis=0)


class Padding(object):
    def __init__(self, pad_token="[PAD]", max_length=20):
        self._pad_token = pad_token
        self._max_length = max_length

    def __call__(self, sample):
        padded_tokens = sample.tokens + [self._pad_token] * (self._max_length - len(sample.tokens))
        return {"padded_tokens": padded_tokens, "intent": sample.intents}


class ToTensor(object):
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
        sub_sample = [True] * len(self._data)
        for j, id in enumerate(self._data.index):
            _, _, sub_id = id.split("-")
            if sub_id == "full":
                continue
            elif int(sub_id) < self._initial_prefix or (int(sub_id) - self._initial_prefix) % self._incremental_size != 0:
                sub_sample[j] = False

        return self._data.index[np.array(sub_sample)]

    def __len__(self):
       return len(self._data)

    def __getitem__(self, idx):
        if self._transform is not None:
            return self._transform(self._data.loc[idx, :])
        else:
            return dict(tokens=self._data.tokens[idx], intent=self._data.intent[idx])


#  class IncrementalLoader(object):

    #  """Data loader for incremental data."""

    #  def __init__(self, file, name, config,):
        #  """Initialize the loader instance.

        #  :file: path to data file
        #  :name: dataset name
        #  :config: config file

        #  """

        #  self._file = file
        #  self._name = name
        #  self._config = config


#  idr.prefix_set.keys()
#  len(idr.prefix_intent)
#  len(idr.prefix_set)
#  len(idr.prefix_set["size_3"]["what is the"])
#  idr.prefix_set["size_3"]
#  idr.prefix_intent.keys()
#  len(idr.prefix_intent["size_3"]["what is the"])
#  Counter(idr.prefix_intent["what is"])

#  plt.hist(idr.prefix_set["size_3"]["what is the"], rwidth=0.5)
#  plt.xticks(rotation=45, ha="right")
#  plt.hist(idr.prefix_intent["size_3"]["show me the"], rwidth=0.75)
#  plt.xticks(rotation=60)

#  def entropy(prob):
    #  return np.sum([-p*np.log2(p) for p in prob])


#  entropy([1/3, 1/3, 1/3])

#  target = idr.prefix_intent["size_2"]["what is"]
#  ct = Counter(target)
#  idr.all_intent
#  names = list(idr.all_intent.keys())
#  values = list(idr.all_intent.values())
#  sub_values = {}
#  for name in names:
    #  if name in ct:
        #  sub_values[name] = ct[name]
    #  else:
        #  sub_values[name] = 0

#  from matplotlib.ticker import PercentFormatter
#  fig, ax = plt.subplots(1, 2, figsize=(18,12))
#  plt.xticks(rotation=45, ha="right")
#  plt.xticks(rotation=45, ha="right")
#  ax[0].bar(names, values)
#  ax[0].yaxis.set_major_formatter(PercentFormatter(xmax=sum(values), decimals=0, is_latex=True))
#  ax[0].set_xticklabels(names, rotation=45, ha="right")
#  ax[1].bar(names, sub_values.values())
#  #  ax[1].hist(target, align="right", rwidth=0.75)
#  ax[1].yaxis.set_major_formatter(PercentFormatter(xmax=len(target), decimals=0, is_latex=True))
#  ax[1].set_xticklabels(names, rotation=45, ha="right")
#  ax[0].get_xaxis().__dict__
#  ax[0].get_xaxis()["majorTicks"][0]
#  ax[0]._x
#  ax[0].xaxis.get_majorTicks()
#  ax[0].get_xticklabels()[1]

#  ax.set_xticklabels()
#  plt.show()
#  plt.xticks(rotation=45, ha="right", major_formatter=PercentFormatter)
#  plt.xticklabels
