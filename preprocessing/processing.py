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
from collections import Counter, defaultdict
import json
import numpy as np
import pandas as pd

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, "preprocessing"))
sys.path.append(os.path.join(cwd, 'preprocessing/raw_data/top-dataset-semantic-parsing'))

from tree import Tree, Intent, Slot

raw_data_dir = os.path.join(cwd, 'preprocessing/raw_data')
data_file = ['train', 'test', 'dev']


# preprocess ATIS data
atis_dir = 'atis_resplit'
atis_out = os.path.join(cwd, 'data/atis')
intent_vocab = defaultdict(set)
slot_vocab = defaultdict(set)

for file in data_file:

    atis_df = pd.read_csv(os.path.join(raw_data_dir, atis_dir, f'atis.{file}.csv'))

    token_list = atis_df.tokens.apply(lambda s: s.split(' '))
    tag_list = atis_df.slots.apply(lambda s: s.split(' '))

    data_points = []
    for i in range(len(atis_df)):
        intents = atis_df.intent[i].split("#")
        intent_vocab[file].update(set(intents))
        slot_vocab[file].update(set(np.unique(tag_list[i])))

        id = atis_df.id[i]
        token_list[i][0] = "<BOS>"
        token_list[i][-1] = "<EOS>"
        token_tag_pair = [f"{token} {tag}" for token, tag in zip(token_list[i], tag_list[i])]

        for j in range(1, len(token_list[i])):
            partial_id = id + f"-{j}"
            if token_tag_pair[j].startswith("<EOS>"):
                partial_id = id + "-full"
            writable = '\n'.join([partial_id] + token_tag_pair[:j + 1] + [atis_df.intent[i]])
            data_points.append(writable)


    with open(os.path.join(atis_out, f"{file}.txt"), 'w') as f:
        f.write('\n\n'.join(data_points))

with open(os.path.join(atis_out, "intent.vocab"), 'w') as f:
    f.write('\n'.join(list(intent_vocab['train'])))
with open(os.path.join(atis_out, "slot.vocab"), 'w') as f:
    f.write('\n'.join(list(slot_vocab['train'])))


# preprocess FB TOP semantic parsing data
top_dir = 'top-dataset-semantic-parsing'
top_out = os.path.join(cwd, 'data/top')

intent_vocab = defaultdict(set)
slot_vocab = defaultdict(set)

for file in data_file:
    data_points = []
    with open(os.path.join(raw_data_dir, top_dir, f"{file}.tsv"), 'r') as f:
        i = 0
        for line in f:
            i += 1
            id = f"{file}-{i:05d}"
            _, query, top = line.split('\t')
            tokens = query.split(' ')
            num_tokens = len(tokens)
            tree_repr = Tree(top.rstrip())

            all_intents = []
            bio_tags = ["O"] * num_tokens
            intent_triggers = defaultdict(list)

            processed_index = 0
            comb = False
            for member in tree_repr.root.list_nonterminals():
                if isinstance(member, Intent):
                    intent = member.label[3:]
                    if intent == "COMBINE":
                        comb = True
                        #  print(member)
                        continue
                    else:
                        all_intents.append(intent)
                        span = member.get_token_indices()
                        for x in range(span[0], num_tokens):
                            intent_triggers[x+1].append(intent)

                elif isinstance(member, Slot):
                    slot_name = member.label[3:]
                    if slot_name == "COMBINE":
                        comb = True
                        continue
                    span = member.get_token_indices()
                    if span[0] > processed_index:
                        bio_tags[span[0]] = 'B-' + slot_name
                        if len(span) > 1:
                            bio_tags[span[1] : span[-1] + 1] = ['I-'+ slot_name] * (len(span) - 1)
                        processed_index = span[-1]

            tokens = ["<BOS>"] + tokens + ["<EOS>"]
            bio_tags = ["O"] + bio_tags + ["O"]
            assert (num_tokens + 2 == len(bio_tags)), 'BIO tags of wrong size'
            intent_vocab[file].update(set(all_intents))
            slot_vocab[file].update(set(np.unique(bio_tags)))


            token_tag_pair = [f"{token} {tag}" for token, tag in zip(tokens, bio_tags)]
            for j in range(1, len(tokens)):
                partial_id = id + f"-{j}"
                cum_intent = '#'.join(intent_triggers[j])

                if token_tag_pair[j].startswith("<EOS>"):
                    partial_id = id + "-full"
                    cum_intent = '#'.join(intent_triggers[j - 1])

                writable = '\n'.join([partial_id] + token_tag_pair[:j+1] + [cum_intent])
                data_points.append(writable)

        with open(os.path.join(top_out, f"{file}.txt"), 'w') as f_out:
            f_out.write('\n\n'.join(data_points))

with open(os.path.join(top_out, "intent.vocab"), 'w') as f:
    f.write('\n'.join(list(intent_vocab['train'])))
with open(os.path.join(top_out, "slot.vocab"), 'w') as f:
    f.write('\n'.join(list(slot_vocab['train'])))


# preprocess FB multilingual semantic parsing data
fbml_dir = "multilingual_task_oriented_dialog_slotfilling"
fbml_out = os.path.join(cwd, "data/fbml")
intent_vocab = defaultdict(set)
slot_vocab = defaultdict(set)

header = ["intent", "slots", "query", "lang", "annotation"]
for file in data_file:
    fbml_df = pd.read_csv(os.path.join(raw_data_dir, fbml_dir, f"{file}-en.tsv"), sep='\t', header=None, names=header)
    fbml_df = fbml_df.drop_duplicates()
    data_points = []

    intent_vocab[file].update(set(fbml_df.intent))

    i = 0
    for dp in fbml_df.itertuples():
        i += 1
        id = f"{file}-{i:05d}"
        annotation = json.loads(dp.annotation)["tokenizations"][0]
        tokens = annotation["tokens"]
        span = annotation["tokenSpans"]
        num_tokens = len(tokens)

        bio_tags = ["O"] * num_tokens
        if not pd.isnull(dp.slots):
            slots = dp.slots.split(",")
            processed_index = 0

            s, e, name = slots[processed_index].split(":")
            tag_begin = True
            j = 0
            while j < num_tokens:
                if span[j]['start'] > int(e):
                    if processed_index < len(slots) - 1:
                        processed_index += 1
                        s, e, name = slots[processed_index].split(":")
                        tag_begin = True
                        continue
                    else:
                        break
                elif span[j]['start'] + span[j]['length'] < int(s):
                    pass
                elif span[j]['start'] >= int(s) and tag_begin:
                    bio_tags[j] = "B-" + name
                    tag_begin = False
                else:
                    bio_tags[j] = "I-" + name
                j += 1

        tokens = ["<BOS>"] + tokens + ["<EOS>"]
        bio_tags = ["O"] + bio_tags + ["O"]
        slot_vocab[file].update(set(bio_tags))
        token_tag_pair = [f"{token} {tag}" for token, tag in zip(tokens, bio_tags)]
        for j in range(1, len(tokens)):
            partial_id = id + f"-{j}"

            if token_tag_pair[j].startswith("<EOS>"):
                partial_id = id + "-full"

            writable = '\n'.join([partial_id] + token_tag_pair[:j+1] + [dp.intent])
            data_points.append(writable)

    with open(os.path.join(fbml_out, f"{file}.txt"), 'w') as f_out:
        f_out.write('\n\n'.join(data_points))

with open(os.path.join(fbml_out, "intent.vocab"), 'w') as f:
    f.write('\n'.join(list(intent_vocab['train'])))
with open(os.path.join(fbml_out, "slot.vocab"), 'w') as f:
    f.write('\n'.join(list(slot_vocab['train'])))



# preprocess SNIPS benchmark data
snips_dir = "nlu-benchmark/2017-06-custom-intent-engines"
snips_out = os.path.join(cwd, "data/snips")
intent_vocab = defaultdict(set)
slot_vocab = defaultdict(set)

domains = os.listdir(os.path.join(cwd, raw_data_dir, snips_dir))
snips_data = defaultdict(list)

for intent in domains:
    if not intent.endswith('.md'):
        train_data = json.load(open(os.path.join(cwd, raw_data_dir, snips_dir, f"{intent}/train_{intent}_full.json"), 'rb'))
        dev_data = json.load(open(os.path.join(cwd, raw_data_dir, snips_dir, f"{intent}/validate_{intent}.json"), 'rb'))
        intent_vocab["train"].add(intent)

        count = len(train_data[intent])
        test_index = set(np.random.choice(count, 100, replace=False))
        for i, v in enumerate(train_data[intent]):
            v["intent"] = intent
            if i in test_index:
                snips_data["test"].append(v)
            else:
                snips_data["train"].append(v)

        for v in dev_data[intent]:
            v["intent"] = intent
            snips_data["dev"].append(v)

for file in data_file:
    data_points = []
    i = 0
    for v in snips_data[file]:
        i += 1
        id = f"{file}-{i:05d}"

        data = v["data"]
        tokens = []
        bio_tags = []
        for dic in data:
            text = [t.encode('utf-8', 'ignore').decode() for t in dic["text"].strip().split(" ")]
            text = [t for t in text if t != ""]
            if "entity" in dic:
                tags = [f"B-{dic['entity']}"] + [f"I-{dic['entity']}"] * (len(text) - 1)
            else:
                tags = ["O"] * len(text)
            tokens.extend(text)
            bio_tags.extend(tags)

        tokens = ["<BOS>"] + tokens + ["<EOS>"]
        bio_tags = ["O"] + bio_tags + ["O"]
        slot_vocab["train"].update(set(bio_tags))
        token_tag_pair = [f"{token} {tag}" for token, tag in zip(tokens, bio_tags)]
        for j in range(1, len(tokens)):
            partial_id = id + f"-{j}"

            if token_tag_pair[j].startswith("<EOS>"):
                partial_id = id + "-full"

            writable = '\n'.join([partial_id] + token_tag_pair[:j+1] + [v["intent"]])
            data_points.append(writable)

    with open(os.path.join(snips_out, f"{file}.txt"), 'w') as f_out:
        f_out.write('\n\n'.join(data_points))

with open(os.path.join(snips_out, "intent.vocab"), 'w') as f:
    f.write('\n'.join(list(intent_vocab['train'])))
with open(os.path.join(snips_out, "slot.vocab"), 'w') as f:
    f.write('\n'.join(list(slot_vocab['train'])))
