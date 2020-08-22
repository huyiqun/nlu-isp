import os
import sys
import pathlib
import argparse
import shlex
from typing import List
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (12, 8), "font.size": 20})

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from src.logging_util import ColoredLog
from src.loader import IncrementalDataReader, IncrementalDataset, Padding, ToTensor

try:
    print(__file__)
except NameError:
    __file__ = "/home/agneshu_google_com/nlu-isp/src/model.py"

source_root = pathlib.Path(__file__).parent
data_root = os.path.join(pathlib.Path(__file__).parents[1], "data")

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_log", type=bool, default=False)
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--wandb_log", type=bool, default=False)
parser.add_argument("--cpu", action="store_true", default=False)
# Training parameters.
parser.add_argument("--init_prefix", "-ip", type=int, default=1)
parser.add_argument("--incre_size", "-is", type=int, default=1)

# Training parameters.
parser.add_argument("--data", "-d", type=str, default="atis")
parser.add_argument("--file", "-f", type=str, default="train.txt")
parser.add_argument("--verbose", "-v", action="count", default=0)
parser.add_argument("--save_dir", "-s", type=str, default="save")
parser.add_argument("--init_prefix", "-ip", type=int, default=1)
parser.add_argument("--incre_size", "-is", type=int, default=1)

parser.add_argument("--random_state", "-rs", type=int, default=0)
parser.add_argument("--num_epoch", "-ne", type=int, default=10)
parser.add_argument("--batch_size", "-bs", type=int, default=16)
parser.add_argument("--l2_penalty", "-lp", type=float, default=1e-6)
parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
parser.add_argument("--dropout_rate", "-dr", type=float, default=0.4)
parser.add_argument("--intent_forcing_rate", "-ifr", type=float, default=0.9)
parser.add_argument("--differentiable", "-d", action="store_true", default=False)
parser.add_argument("--slot_forcing_rate", "-sfr", type=float, default=0.9)

# model parameters.
parser.add_argument("--word_embedding_dim", "-wed", type=int, default=64)
parser.add_argument("--encoder_hidden_dim", "-ehd", type=int, default=256)
parser.add_argument("--intent_embedding_dim", "-ied", type=int, default=8)
parser.add_argument("--slot_embedding_dim", "-sed", type=int, default=32)
parser.add_argument("--slot_decoder_hidden_dim", "-sdhd", type=int, default=64)
parser.add_argument("--intent_decoder_hidden_dim", "-idhd", type=int, default=64)
parser.add_argument("--attention_hidden_dim", "-ahd", type=int, default=1024)
parser.add_argument("--attention_output_dim", "-aod", type=int, default=128)

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
arg_string = "-vvvv"
arg_string = ""
args = parser.parse_args(shlex.split(arg_string))

if args.wandb_log:
    import wandb
    wandb.init(project="test", config=args)
    config = wandb.config
else:
    config = args

VERBOSE = config.verbose
logger = ColoredLog(__file__, verbose=VERBOSE)
device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")

idr = IncrementalDataReader(config.data, config.file, full_utt_only=False)
idr.dataset
idr.all_intent
idr.dataset.partial
idr.slot_vocab
idr.intent_vocab

def print_tokenizer_special(tokenizer):
    table = np.transpose([tokenizer.all_special_ids, tokenizer.all_special_tokens])
    logger.info(table, header=["id", "token"])

tokenizer = BertTokenizer("data/atis/token.vocab", bos_token="<BOS>", eos_token="<EOS>")
print_tokenizer_special(tokenizer)

x = IncrementalDataset(idr.dataset, transform=transforms.Compose([Padding(max_length=idr.max_length), ToTensor(tokenizer)]))
a = DataLoader(x, batch_size=10, sampler=SubsetRandomSampler(x.sampler))
for i, s in enumerate(a):
    print(i)
    print(s)




tokenizer.ids_to_tokens[0]
tokenizer.convert_ids_to_tokens
tokenizer.vocab.keys()
tokenizer.build_inputs_with_special_tokens([95, 209], [95, 209])
tokenizer.pretrained_vocab_files_map

tokenizer.encode("<BOS> I like tea")
s = idr.dataset.tokens[0]
tokenizer.decode(tokenizer.encode(" ".join(s)))
encoded_tensor = torch.Tensor([tokenizer.encode(idr.dataset.partial[j]) for j in range(len(idr.dataset.partial))])
encoded_tensor = torch.Tensor(tokenizer.encode(idr.dataset.partial[0]))

model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
model.eval()
with torch.no_grad():
    outputs = model(encoded_tensor)
    hidden_states = outputs[2]


data_root = os.path.join(pathlib.Path(__file__).parents[1], "data")
os.listdir(data_root)
padding = Padding()
p = padding(idr.dataset.loc["train-00001-1", :])
tt = ToTensor()
tt(p)

x = IncrementalDataset(idr.dataset[:30], initial_prefix=config.init_prefix, incremental_size=config.incre_size, transform=transforms.Compose([Padding(max_length=50), ToTensor()]))
x.sampler
x["train-00001-1"]
isinstance(idr.dataset.tokens["train-00001-1"], list)

def my_collate(batch):
    #  return [len(dp["tokens"]) for dp in batch]
    return torch.Tensor([dp["tokens"] for dp in batch])

a = DataLoader(x, batch_size=10, sampler=SubsetRandomSampler(x.sampler), shuffle=False, collate_fn=my_collate)
for i, s in enumerate(a):
    print(i)
    print(s)



tokenizer = BertTokenizer("data/atis/token.vocab", bos_token="<BOS>", eos_token="<EOS>", model_max_len=50)
tokenizer.prepare_for_model(tokenizer.encode(y), return_tensors="pt")

tokenizer.SPECIAL_TOKENS_ATTRIBUTES
tokenizer.encode(y)
tokenizer.encode_plus(y)
y = "<BOS> embedding what is the flight number <EOS>"
ids = tokenizer.encode_plus
tokenizer.decode(tokenizer.encode(y))
tokenizer.save_pretrained("data/atis/save")
tokenizer.save_vocabulary("data/atis/save/saved")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", bos_token="<BOS>", eos_token="<EOS>")
tokenizer.tokenize("i like tea")
special_tokens = {"bos_token": "<BOS>", "eos_token": "<EOS>"}
tokenizer.add_special_tokens(special_tokens)

tokenizer.bos_token_id
tokenizer.eos_token_id
tokenizer.all_special_ids

tokenizer.special_tokens_map
tokenizer.additional_special_tokens
y = "<BOS> I like embeddings <EOS> [SEP] i like tea"
z = tokenizer.encode(y)
tokenizer.convert_ids_to_tokens(z)
tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(z))


tokenizer.encode("embeddings embedding")
tokenizer.encode("i like tea")
tokenizer.encode("i like tea")
tokenizer.decode(tokenizer.encode("embeddings embedding"))

tokenizer.get_special_tokens_mask([100,101,102],[1,2,3])
tokenizer.get_special_tokens_mask([100,101,102,1,2,3])

tokenizer("s")
from transformers import BertTokenizerFast
t1 = BertTokenizerFast.from_pretrained("bert-base-uncased", bos_token="<BOS>", eos_token="<EOS>")

t1.tokenize("<BOS> I like embeddings <EOS> [SEP] i like tea")
t1.special_tokens_map
y = t1.encode("<BOS> I like embeddings <EOS> [SEP] i like tea")
t1.create_token_type_ids_from_sequences(y)

t1("abd")
t1.covert_ids_to_tokens(y)
type(t1)


embedding_layer = nn.Embedding(idr.vocab_size, embedding_dim=64)
enc = tokenizer.encode(tokenizer.tokenize(idr.dataset.partial[3]))
a = embedding_layer(torch.LongTensor(enc))
a.shape


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        sz = 5
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

