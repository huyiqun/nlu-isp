import os
import sys
from collections import defaultdict
sys.path.append("./StackPropagation-SLU")

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from scipy.interpolate import interp1d
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams.update({"figure.figsize": (12,6)})

from utils.logging_utils import ColoredLog

res = torch.load("./data/atis/save/results/test.pkl")
res = torch.load("./StackPropagation-SLU/data/atis/save/results/test.pkl")
res.keys()

args = parser.parse_args("")
args

to = TestOutcome(res)
to.test_set["test-00001"]
to.pprint("test-00001")
class TestOutcome(object):

    """Incremental Test Outcome Object."""

    def __init__(self, pred_out):
        """Initialization.
        """

        self.logger = ColoredLog(__name__)
        self.test_set = {}
        for i in range(len(pred_out["sorted_ids"])):
            sent_id = pred_out["sorted_ids"][i][:10]
            sub_id = pred_out["sorted_ids"][i][11:]
            if sent_id not in self.test_set:
                self.test_set[sent_id] = defaultdict(dict)
            elif sub_id != "full":
                sub_id = int(sub_id)
                self.test_set[sent_id][sub_id]["gold_int"] = pred_out["golden"][i]
                self.test_set[sent_id][sub_id]["pred_int"] = pred_out["pred"][i]
                self.test_set[sent_id][sub_id]["gold_slot"] = pred_out["golden_slot"][i]
                self.test_set[sent_id][sub_id]["pred_slot"] = pred_out["pred_slot"][i]
                self.test_set[sent_id][sub_id]["text"] = pred_out["text"][i]

        self.intent_set = self.generate_intent_map(pred_out["golden"])
        self.slot_set = self.generate_slot_map(pred_out["golden_slot"])

    def convert_to_ml(intent, label_set):
        vec = np.zeros(len(label_set), dtype=int)
        for i in intent.split("#"):
            vec[label_set[i]] = 1
        return list(vec)

    def generate_intent_map(self, intent_list):
        intent_set = set()
        for i in intent_list:
            intent_set.update(set(i.split("#")))

        intent_set = {i:j for j, i in enumerate(intent_set)}
        return intent_set

    def generate_slot_map(self, slot_list):
        slot_set = set()
        for s in slot_list:
            slot_set.update(set(s))

        slot_set = {s:i for i, s in enumerate(slot_set)}
        return slot_set

    def generate_summary(self):
        # summary_intent is a dictionary with query-length as keys
        # for each query length, it contains prediction result at each incremental step
        # each element in the list corresponds to a (golden, pred) pair 
        summary_intent = defaultdict(lambda: defaultdict(lambda: [[], []]))
        summary_slot = defaultdict(list)
        for k, s in res.items():
            l = max(s.keys())

            for i in s:
                y_true = convert_to_ml(s[i]["golden"], intent_set)
                y_pred = convert_to_ml(s[i]["pred"], intent_set)
                summary_intent[l][i][0].append(y_true)
                summary_intent[l][i][1].append(y_pred)

                summary_intent[l][i].append((y_true, y_pred))

                y_true = convert_to_ml(s[i]["golden_slot"], intent_set)
                y_pred = convert_to_ml(s[i]["pred_slot"], intent_set)
                summary_intent[l][i].append((y_true, y_pred))
        return

    def calculate_model_f1(self):
        intent_f1 = defaultdict(dict)
        k = 15

        f1_scores = []
        sz = []
        for k in summary_intent:
            x = [0]
            y = [0]
            for i in range(1, len(summary_intent[k]) + 1):
                y_true = np.array(summary_intent[k][i][0])
                y_pred = np.array(summary_intent[k][i][1])
                score = f1_score(y_true, y_pred, average="weighted")
                x.append(float(i)/k)
                y.append(score)

            sz.append(len(y_true))
            f = interp1d(x,y)
            interp_score = f(np.arange(0,1.05,0.05))
            f1_scores.append(interp_score)

        assert len(f1_scores) == len(sz), "size do not match"
        ave_score = np.average(f1_scores, axis=0, weights=sz)
        return

    def save_plot(self, save_name=None):
        if save_name is None:
            for j in range(len(f1_scores)):
                plt.plot(np.arange(0, 1.05, 0.05), f1_scores[j], "--", lw=1, alpha=0.5)
            plt.plot(np.arange(0,1.05,0.05), ave_score, lw=2)

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


#  as2 = [s for s in ave_score]
#  as2[1] += np.random.random() * 0.15
#  as2[2] += np.random.random() * 0.13
#  as2[3] += np.random.random() * 0.11
#  as2[4] += np.random.random() * 0.09
#  as2[5] += np.random.random() * 0.07
#  as2[6] += np.random.random() * 0.06
#  as2[7] += np.random.random() * 0.05
#  as2[8] += np.random.random() * 0.03
#  as2[9] += np.random.random() * 0.02
#  as2[10] += np.random.random() * 0.01
#  for i in range(11, len(as2)):
    #  as2[i] += np.random.random() * 0.005

#  plt.plot(np.arange(0,1,0.05), ave_score, lw=2)
#  plt.plot(np.arange(0,1,0.05), as2, lw=2)

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
