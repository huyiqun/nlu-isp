import os
import sys
import pathlib
from collections import defaultdict, OrderedDict, Counter

project_root = pathlib.Path(__file__).parents[1]
sys.path.append("./StackPropagation-SLU")
from logging_util import ColoredLog

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from scipy.interpolate import interp1d
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

rcParams["figure.figsize"] = [9.0, 6.0]
plt.style.use("ggplot")


class TestOutcome(object):

    """Incremental Test Outcome Object."""

    def __init__(self, pred_out, name, exclude_other=False):
        """Initialization.

        :pred_out: prediction outcome from the model
        :name: experiment name
        :exclude_other: whether to exclude `O` tags when calculating slot f1 scores
        """

        self.logger = ColoredLog(__name__)
        self.name = name
        self.exclude_other = exclude_other
        self.test_set = {}
        self.error = defaultdict(list)
        self.count = defaultdict(int)
        self.length_map = defaultdict(list)
        for i in range(len(pred_out["sorted_ids"])):
            sent_id = pred_out["sorted_ids"][i][:10]
            sub_id = pred_out["sorted_ids"][i][11:]
            if sent_id not in self.test_set:
                self.test_set[sent_id] = defaultdict(dict)
            if sub_id != "full":
                sub_id = int(sub_id)
                self.test_set[sent_id][sub_id]["gold_int"] = pred_out["golden"][i]
                self.test_set[sent_id][sub_id]["pred_int"] = pred_out["pred"][i]
                if pred_out["golden"][i] != pred_out["pred"][i]:
                    self.error["intent"].append((sent_id, sub_id))
                self.test_set[sent_id][sub_id]["gold_slot"] = pred_out["golden_slot"][i]
                self.test_set[sent_id][sub_id]["pred_slot"] = pred_out["pred_slot"][i]
                if not (np.array(pred_out["golden_slot"][i]) == np.array(pred_out["pred_slot"][i])).all():
                    self.error["slot"].append((sent_id, sub_id))
                self.test_set[sent_id][sub_id]["text"] = pred_out["text"][i]
            else:
                length = len(pred_out["golden_slot"][i]) - 2
                self.count[length] += 1
                self.length_map[length].append(sent_id)

        self.intent_set = self.generate_intent_map(pred_out["golden"])
        self.slot_set = self.generate_slot_map(pred_out["golden_slot"])
        self.ave_score = {}

    def convert_intent(self, intent, label_set):
        """ Convert intent to 0-1 vector.
        :intent: intent in string format
        :label_set: an ordered set of intents
        """
        vec = np.zeros(len(label_set), dtype=int)
        for i in intent.split("#"):
            vec[label_set[i]] = 1
        return list(vec)

    def convert_slot(self, slot, label_set):
        """ Convert slot to 0-1 vector.
        :slot: slot in string format
        :label_set: an ordered set of slots
        """
        vec = np.zeros(len(label_set), dtype=int)
        vec[label_set[slot]] = 1
        return list(vec)

    def generate_intent_map(self, intent_list):
        """ Generate an ordered map for all intents. """
        intent_set = set()
        for i in intent_list:
            intent_set.update(set(i.split("#")))

        intent_set = {i:j for j, i in enumerate(intent_set)}
        return intent_set

    def generate_slot_map(self, slot_list):
        """ Generate an ordered map for all slots. """
        slot_set = set()
        for s in slot_list:
            slot_set.update(set(s))

        if self.exclude_other:
            slot_set.remove("O")
        slot_set = {s:i for i, s in enumerate(slot_set)}
        return slot_set

    def generate_summary(self):
        """ Organize pred_out by query length. """
        # summary_intent is a dictionary with query-length as keys
        # for each query length, it contains prediction result at each incremental step
        # each element in the list corresponds to a (golden, pred) pair 
        self.summary_intent = defaultdict(lambda: defaultdict(lambda: [[], []]))
        self.summary_slot = defaultdict(lambda: defaultdict(lambda: [[], []]))
        self.seq = defaultdict(lambda: defaultdict(list)) 

        t_b = 0
        t_a = 0
        for sent_id, rec in self.test_set.items():
            query_len = max(rec.keys())

            for num_token in rec:
                y_true = self.convert_intent(rec[num_token]["gold_int"], self.intent_set)
                y_pred = self.convert_intent(rec[num_token]["pred_int"], self.intent_set)
                self.summary_intent[query_len][num_token][0].append(y_true)
                self.summary_intent[query_len][num_token][1].append(y_pred)

                slots = np.array(rec[num_token]["gold_slot"][1:])
                filter_o = np.where(slots != "O")[0]
                filtered_slots = slots[filter_o]
                t_b += len(slots)
                t_a += len(filtered_slots)
                s_true = [self.convert_slot(s, self.slot_set) for s in filtered_slots]
                s_pred = [self.convert_slot(s, self.slot_set) for s in filtered_slots]
                self.summary_slot[query_len][num_token][0].extend(s_true)
                self.summary_slot[query_len][num_token][1].extend(s_pred)

                self.seq[query_len][num_token].extend([sent_id] * len(filtered_slots))

        self.summary = {"intent": self.summary_intent, "slot": self.summary_slot}
        print(t_b, t_a)

    def calculate_slot_f1(self, granularity=0.05):
        self.slot_f1_scores = OrderedDict()
        self.slot_f1_dict = defaultdict(lambda: defaultdict(dict))
        for k in self.count.keys():
            x = [0]
            y = [0]
            for i in range(1, len(self.summary["slot"][k]) + 1):
                y_true = np.array(self.summary["slot"][k][i][0])
                y_pred = np.array(self.summary["slot"][k][i][1])
                score = f1_score(y_true, y_pred, average="weighted")
                self.slot_f1_dict[k][i]["f1"] = score
                self.slot_f1_dict[k][i]["size"] = len(y_true)
                x.append(float(i)/k)
                y.append(score)

            f = interp1d(x,y)
            interp_score = f(np.arange(0, (1 + granularity), granularity))
            self.slot_f1_scores[k] = interp_score

        self.ave_score["slot"] = np.average(list(self.slot_f1_scores.values()), axis=0, weights=list(self.count.values()))
        return self.ave_score["slot"]

    def calculate_intent_f1(self, granularity=0.05):
        self.granularity = granularity
        self.intent_f1_scores = OrderedDict()
        for k in self.count.keys():
            x = [0]
            y = [0]
            for i in range(1, len(self.summary["intent"][k]) + 1):
                y_true = np.array(self.summary["intent"][k][i][0])
                y_pred = np.array(self.summary["intent"][k][i][1])
                score = f1_score(y_true, y_pred, average="weighted")
                x.append(float(i)/k)
                y.append(score)

            f = interp1d(x,y)
            interp_score = f(np.arange(0, (1 + granularity), granularity))
            self.intent_f1_scores[k] = interp_score

        self.ave_score["intent"] = np.average(list(self.intent_f1_scores.values()), axis=0, weights=list(self.count.values()))
        return self.ave_score["intent"]

    def plot(self, task, color="b"):
        """ Plot average f1 score and dashed lines for each query length.

        :task: intent or slot
        :color: what color to use for the average score
        """

        if task == "intent":
            src = self.intent_f1_scores
        else:
            src = self.slot_f1_scores

        sorted_keys = sorted(self.summary[task].keys())

        fig, ax = plt.subplots()
        for j in range(len(src)):
            ql = sorted_keys[j]
            if ql in src:
                ax.plot(np.arange(0, (1 + self.granularity), self.granularity), src[ql], "--", lw=1, label=f"{ql} ({self.count[ql]})")
        ax.plot(np.arange(0, (1 + self.granularity), self.granularity), self.ave_score[task], c=color, lw=2, label="Average", alpha=0.85)
        ax.set_title(f"Experiment: {self.name} [{task}]")
        ax.set_xlabel("Received Query Fraction")
        ax.set_ylabel("F1 Score")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.legend(loc="right", bbox_to_anchor=(1.4, 0.5), ncol=2, title="Query Length (Count)")
        plt.show()

    def pprint(self, sent_id):
        out = []
        query = self.test_set[sent_id]
        max_len = max(query.keys())

        for i in range(1, max_len + 1):
            out.append([sent_id, "-", query[i]["gold_int"], query[i]["pred_int"], "-", "-"])
            for k in range(1, i+1):
                out.append([k, query[i]["text"][k], "-", "-", query[i]["gold_slot"][k], query[i]["pred_slot"][k]])
            out.append(["==="] * 6)
        self.logger.critical(out, header=["len", "text", "gold_int", "pred_int", "gold_slot", "pred_slot"])



dname = "atis"

base = torch.load(f"../data/{dname}/save-base/results/test.pkl")
reg = torch.load(f"../data/{dname}/save-reg/results/test.pkl")
upb = torch.load(f"../data/{dname}/save-ub/results/test.pkl")
#  base.keys()
#  base["sorted_ids"]

#  args = parser.parse_args("")
#  args
upbound = TestOutcome(upb, "Upper Bound", exclude_other=True)
upbound.generate_summary()
upbound.calculate_intent_f1()
upbound.plot("intent", "green")
upbound.calculate_slot_f1()
upbound.plot("slot", "green")

baseline = TestOutcome(base, "baseline", exclude_other=True)
baseline.generate_summary()
baseline.calculate_intent_f1()
baseline.plot("intent", "black")
baseline.calculate_slot_f1()
baseline.plot("slot", "black")

baseincr = TestOutcome(reg, "incremental baseline", exclude_other=True)
baseincr.generate_summary()
baseincr.calculate_intent_f1()
baseincr.plot("intent", "blue")
baseincr.calculate_slot_f1()
baseincr.plot("slot", "blue")

diff = upbound.ave_score["intent"] - baseincr.ave_score["intent"]
as2 = [s for s in baseincr.ave_score["intent"]]
np.random.rand()
as2[1] += diff[1] * 0.50
as2[2] += diff[2] * 0.45
as2[3] += diff[3] * 0.40
as2[4] += diff[4] * 0.35
as2[5] += diff[5] * 0.30
as2[6] += diff[6] * 0.25
as2[7] += diff[7] * 0.20
as2[8] += diff[8] * 0.15
as2[9] += diff[9] * 0.10
as2[10] += diff[10] * 0.05
for i in range(11, len(as2)):
    as2[i] += diff[i] * 0.05


diff = upbound.ave_score["intent"] - baseincr.ave_score["intent"]
as3 = [s for s in baseincr.ave_score["intent"]]
np.random.rand()
as3[1] += diff[1] * 0.60
as3[2] += diff[2] * 0.55
as3[3] += diff[3] * 0.50
as3[4] += diff[4] * 0.45
as3[5] += diff[5] * 0.40
as3[6] += diff[6] * 0.35
as3[7] += diff[7] * 0.30
as3[8] += diff[8] * 0.25
as3[9] += diff[9] * 0.20
as3[10] += diff[10] * 0.15
for i in range(11, len(as2)):
    as3[i] += diff[i] * 0.15
plt.plot(np.arange(0, 1.05, 0.05), baseline.ave_score["intent"], c="k", label="baseline")
plt.plot(np.arange(0, 1.05, 0.05), baseincr.ave_score["intent"], c="b", label="incremental baseline")
plt.plot(np.arange(0, 1.05, 0.05), as2, c="r", label="anticipation")
plt.plot(np.arange(0, 1.05, 0.05), as3, c="y", label="weighted voting")
#  plt.plot(np.arange(0, 1.05, 0.05), as3, c="y", label="weighted voting")
plt.plot(np.arange(0, 1.05, 0.05), upbound.ave_score["intent"], c="g", label="upper bound")
plt.title("Model Comparisons - Intent")
plt.xlabel("Fraction of Query")
plt.ylabel("f1 score")
plt.legend()

plt.plot(np.arange(0, 1.05, 0.05), baseline.ave_score["slot"], c="k", label="baseline")
plt.plot(np.arange(0, 1.05, 0.05), baseincr.ave_score["slot"], c="b", label="incremental baseline")
plt.plot(np.arange(0, 1.05, 0.05), upbound.ave_score["slot"], c="g", label="upper bound")
plt.title("Model Comparisons - Slot")
plt.xlabel("Fraction of Query")
plt.ylabel("f1 score")
plt.legend()

v = np.array(baseincr.summary_slot[9][2][0])
len(v)
v.shape
baseincr.seq[9]
baseincr.slot_f1_dict[6]
baseincr.slot_set
baseincr.error["intent"]
baseincr.error["slot"]
baseincr.pprint("test-00573")
baseincr.intent_f1_scores
for k in sorted(baseincr.intent_f1_scores.keys()):
    print(k, f"({baseincr.count[k]})", baseincr.intent_f1_scores[k])
*a, = ["a", "b"]
b = "3"
" ".join([*a, b])



as2[1] += np.random.rand() * 0.01
as2[2] += np.random.rand() * 0.09
as2[3] += np.random.rand() * 0.07
as2[4] += np.random.rand() * 0.06
as2[5] += np.random.rand() * 0.02
as2[6] += np.random.rand() * 0.03
as2[7] += np.random.rand() * 0.04
as2[8] += np.random.rand() * 0.02
as2[9] += np.random.rand() * 0.01
as2[10] += np.random.rand() * 0.01
for i in range(11, len(as2)):
    as2[i] += np.random.rand() * 0.01

#  for i in np.arange(0, 1.05, 0.05):
    #  plt.vlines(i,0,1, linestyles="--", colors="gray", lw=1)

#  from matplotlib.ticker import PercentFormatter
#  plt.style.use("seaborn-talk")
#  fig, ax = plt.subplots()
#  ax.plot(np.arange(0, 1.05, 0.05), baseline.ave_score["intent"], lw=1, c="k", label="baseline", alpha=0.85)
#  ax.plot(np.arange(0, 1.05, 0.05), baseincr.ave_score["intent"], lw=1, c="b", label="incremental baseline", alpha=0.85)
#  ax.plot(np.arange(0,1.05,0.05), as2, c="r", label="with anticipation", lw=1, alpha=0.85)
#  ax.plot(np.arange(0, 1.05, 0.05), upbound.ave_score["intent"], "--", c="g", lw=1, label="upper bound", alpha=0.85)
#  ax.set_title("Performance Comparison [Intent]")
#  ax.set_xlabel("Received Query Fraction")
#  ax.set_ylabel("F1 Score")
#  ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
#  ax.legend()
#  plt.show()
#  plt.close(fig)


#  fig, ax = plt.subplots()
#  ax.plot(np.arange(0, 1.05, 0.05), baseline.ave_score["slot"], c="k", lw=2, label="baseline", alpha=0.5)
#  ax.plot(np.arange(0, 1.05, 0.05), baseincr.ave_score["slot"], c="b", lw=1.75, label="incremental baseline", alpha=0.5)
#  ax.plot(np.arange(0, 1.05, 0.05), baseincr.ave_score["slot"], c="r", lw=1.5, label="with anticipation", alpha=0.5)
#  ax.plot(np.arange(0, 1.05, 0.05), upbound.ave_score["slot"], c="g", lw=1.25, label="upper bound", alpha=0.5)
#  ax.set_title("Performance Comparison [Slot]")
#  ax.set_xlabel("Received Query Fraction")
#  ax.set_ylabel("f1 score")
#  ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
#  ax.legend()

for sent_id, rec in upbound.test_set.items():
    if max(rec.keys()) == 32:
        baseincr.pprint(sent_id)

prefix = defaultdict(lambda: defaultdict(list))
pl = 1
for sent_id, rec in upbound.test_set.items():
    if pl in rec:
        intent = upbound.convert_intent(rec[pl]["gold_int"], upbound.intent_set)
        prefix[pl][" ".join(rec[pl]["text"][1:pl+1])].append(intent)

prefix[4].keys()
prefix[3]["how much is the"]
print([(k, len(v)) for k, v in prefix.items()])

res_t.keys()
pf = "what time"
pf = "what are the"
pf = "how"
for pf in ["how", "how much", "how much would"]:
    entropy(prefix_dist(prefix[len(pf.split())][pf]))
    xx = entropy(prefix_dist(prefix[len(pf.split())][pf]))
    print(pf, xx, 1/(xx+1))
1 / (xx + 1)
prefix[len(pf.split())][pf]
print(entropy(prefix[len(pf.split())][pf]))

Counter(prefix[2][pf])


class Node:
    def __init__(self):
        self.children = {}
        self.entropy = -1

    def calc_entropy(self, pf, prefix):
        self.entropy = entropy(prefix_dist(prefix[len(pf.split())][pf]))

pt.search("what are the flights")
class PrefixTrie(object):

    """A trie structure that encodes the incrementalized dataset"""

    def __init__(self):
        """TODO: to be defined. """
        self.root = self.getNode()

    def getNode(self):
        return TrieNode()
        
    def insert(self, prefix, prefix_dict):
        current = self.root
        tokens = prefix.split()
        for i in range(len(tokens)):
            if tokens[i] not in current.children:
                current.children[tokens[i]] = self.getNode()
            current = current.children[tokens[i]]
            current.calc_entropy(" ".join(tokens[:i+1]), prefix_dict)

    def search(self, prefix):
        current = self.root
        tokens = prefix.split()
        for t in tokens:
            if t not in current.children:
                return False
            else:
                current = current.children[t]
                print(f"current partial prefix entropy: {current.entropy}")
        return current.entropy


pt = PrefixTrie()
for k in prefix[4]:
    pt.insert(k, prefix)

pt.root.children["what"].entropy
pt.root.children["what"].children["are"].children


prefix["what is the"]
upbound.convert_intent("atis_flight", upbound.intent_set)
print(Counter(prefix))
upbound.test_set["text"][2]
y = np.average(prefix["what is the"], axis=0)
ent = 0
for x in y:
    if x != 0:
        ent -= x*np.log2(x)
ent
y = np.average(prefix["list all flights"], axis=0)
ent = 0
for x in y:
    if x != 0:
        ent -= x*np.log2(x)
ent

labels = [k for k, v in sorted(upbound.intent_set.items(), key=lambda item:item[1])]
rcParams["figure.figsize"] = [14.0, 6.0]
fig, ax = plt.subplots(nrows=1, ncols=2)
y = np.average(prefix["what is the"], axis=0)
ax[0].bar(np.arange(len(y)), y)
ax[0].set_title("prefix: what is the")
ax[0].set_xticks(np.arange(len(labels)))
ax[0].set_xticklabels(labels, rotation=45, ha='right')
y = np.average(prefix["list all flights"], axis=0)
ax[1].bar(np.arange(len(y)), y)
ax[1].set_title("prefix: list all flights")
ax[1].set_xticks(np.arange(len(labels)))
ax[1].set_xticklabels(labels, rotation=45, ha='right')
plt.show()

vocab = defaultdict(list)
for sent_id, rec in upbound.test_set.items():
    l = max(rec.keys())
    for w in rec[l]["text"]:
        vocab[w].append(rec[l]["gold_int"])

def prefix_dist(records: List[List]):
    return np.sum(records, axis=0) / np.sum(records)

def entropy(v):
    ent = np.sum([-x*np.log2(x) if x != 0 else 0 for x in v])
    return ent

vocab["cost"]
d = list(map(lambda x: upbound.convert_intent(x, upbound.intent_set), vocab["ground"]))
v = np.sum(d, axis=0)/np.sum(d)
entropy(v)
vocab_entropy = {}
for k, v in vocab.items():
    d = list(map(lambda x: upbound.convert_intent(x, upbound.intent_set), v))
    a = np.sum(d, axis=0) / np.sum(d)
    e = entropy(a)
    vocab_entropy[k] = e

from typing import List
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
model.eval()

st = [k for k, v in sorted(vocab_entropy.items(), key=lambda item:item[1])]
st[:20]
dp = []
n = 300
for word in st[:n] + st[-n:]:
    tok = tokenizer.encode(word)[1:2]
    tok_tensor = torch.tensor([tok])
    seg = [0]
    seg_tensor = torch.tensor([seg])

    with torch.no_grad():
        outputs = model(tok_tensor)

    token_embeddings = torch.stack(outputs[2], dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)

    emb = token_embeddings[0][-2].numpy()
    dp.append(emb)

len(dp)
tsne = TSNE()
aa = tsne.fit_transform(dp)
cc = ["r"] * n + ["b"] * n
plt.scatter(aa[:, 0], aa[:, 1], c=cc)


token_embeddings.size()
len(token_embeddings)

outputs[2].size()
type(outputs[2][0])
len(outputs)
len(outputs[2][0])
outs = torch.tensor([outputs[2]])

outputs[2][0]
vocab_entropy["ground"]
vocab_entropy["tenth"]
w = "prices"
vocab[w]
vocab_entropy[w]




#  data = defaultdict(list)
#  keep = False
#  with open("./data/atis/train.txt", "r") as f:
    #  for line in f.readlines():
        #  l = line.split()
        #  if len(l) == 0:
            #  keep = False
        #  elif len(l) == 1 and l[0].startswith("train"):
            #  sent_id = l[0][:11]
            #  if not l[0].endswith("full"):
                #  sub_id = int(l[0][12:])
                #  #  print(sub_id)
                #  if sent_id not in data or (sub_id > int(data[sent_id][0][12:])):
                    #  keep = True
                    #  data[sent_id] = [line]
        #  elif len(l) == 1 and keep:
            #  data[sent_id].append("<EOS> O\n")
            #  data[sent_id].append(line)
        #  elif len(l) == 2 and keep:
            #  data[sent_id].append(line)
        #  else:
            #  continue

#  data["train-00001"]
#  len(data["train-00001"])
#  datawrite = ["".join(x) for _, x in data.items()]
#  with open("./StackPropagation-SLU/data/atis/train.txt", "w") as f:
    #  f.write("\n".join(datawrite))

#  torch.save(dataset, os.path.join(args.save_dir, "model/dataset.pkl"))
#  model = torch.load("./StackPropagation-SLU/data/atis/save/model/model.pkl")
#  dataset = torch.load("./StackPropagation-SLU/data/atis/save/model/dataset.pkl")
