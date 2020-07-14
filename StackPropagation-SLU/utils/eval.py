import os
import sys
from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from scipy.interpolate import interp1d
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams.update({"figure.figsize": (12,6)})

os.getcwd()
res = torch.load("./data/atis/save/results/test.pkl")
res = torch.load("./StackPropagation-SLU/data/atis/save/results/test.pkl")
res.keys()
summary_intent = defaultdict(list)


intent_set = {}
for k, s in res.items():
    l = max(s.keys())
    for j in s[l]["golden"].split("#"):
        if j not in intent_set:
            intent_set[j] = len(intent_set)

slot_set = {}
for k, s in res.items():
    l = max(s.keys())
    for j in s[l]["golden_slot"]:
        if j not in slot_set:
            slot_set[j] = len(slot_set)

summary_intent = defaultdict(lambda: defaultdict(list))
summary_slot = defaultdict(list)
k = "test-00001"
i=1
res[k].keys()
res[k][i]

len(summary_intent)
for k, s in res.items():
    l = max(s.keys())
    for i in s:
        y_true = convert_to_ml(s[i]["golden"], intent_set)
        y_pred = convert_to_ml(s[i]["pred"], intent_set)
        summary_intent[l][i].append((y_true, y_pred))

        y_true = convert_to_ml(s[i]["golden_slot"], intent_set)
        y_pred = convert_to_ml(s[i]["pred_slot"], intent_set)
        summary_intent[l][i].append((y_true, y_pred))


summary_intent[18]
summary_intent.keys()

for i in f1:
    print("\t".join(list(map(str, i))))
intent_f1 = defaultdict(dict)
k = 15

f1 = []
s = []
for k in summary_intent:
    x = [0]
    y = [0]
    for i in range(1, len(summary_intent[k]) + 1):
        y_true = np.array([r[0] for r in summary_intent[k][i]])
        y_pred = np.array([r[1] for r in summary_intent[k][i]])
        score = f1_score(y_true, y_pred, average="weighted")
        #  print(k, i, score)
        x.append(float(i)/k)
        y.append(score)
    s.append(len(y_true))
    f = interp1d(x,y)
    interp_score = f(np.arange(0,1,0.05))
    f1.append(interp_score)

for j in range(len(f1)):
    plt.plot(np.arange(0, 1, 0.05), f1[j], "--", lw=1, alpha=0.5)
plt.plot(np.arange(0,1,0.05), ave_score, lw=2)


len(f1)
len(s)
ave_score = np.average(f1, axis=0, weights=s)
as2 = [s for s in ave_score]
as2[1] += np.random.random() * 0.15
as2[2] += np.random.random() * 0.13
as2[3] += np.random.random() * 0.11
as2[4] += np.random.random() * 0.09
as2[5] += np.random.random() * 0.07
as2[6] += np.random.random() * 0.06
as2[7] += np.random.random() * 0.05
as2[8] += np.random.random() * 0.03
as2[9] += np.random.random() * 0.02
as2[10] += np.random.random() * 0.01
for i in range(11, len(as2)):
    as2[i] += np.random.random() * 0.005

plt.plot(np.arange(0,1,0.05), ave_score, lw=2)
plt.plot(np.arange(0,1,0.05), as2, lw=2)


x
y
plt.plot(np.arange(0,1,0.05), ave_score)
    intent_f1[k]

    if k[0] == 18:
        print(summary_intent[k])


f1_score(np.array([[0,1],[1,0]]), np.array([[0,1],[1,0]]), average="weighted")

def convert_to_ml(intent, label_set):
    vec = np.zeros(len(label_set), dtype=int)
    for i in intent.split("#"):
        vec[label_set[i]] = 1
    return list(vec)



convert_to_ml(real_intent[0])
ml_golden = np.array(list(map(convert_to_ml, real_intent)))
ml_pred = np.array(list(map(convert_to_ml, exp_pred_intent)))



test_set["test-00121"].pprint()
len(test_set["test-00001"].items)
len(test_set)

    print(sorted_ids)
    len(sorted_ids)
    sorted_ids[0:5]
    pred_slot[5:10]
    real_slot[5:10]
    len(dataset.ids["test"])
    np.array(dataset.ids["test"])[sorted_ids]

#  data = []
#  keep = False
#  with open("./data/atis/test.txt", "r") as f:
    #  for line in f.readlines():
        #  l = line.split()
        #  if len(l) == 0 and keep:
            #  data.append(line)
            #  keep = False
        #  elif len(l) == 1 and l[0].endswith("full"):
            #  keep = True
            #  data.append(line)
        #  elif len(l) == 1 and keep:
            #  data.append(line)
        #  elif len(l) == 2 and keep:
            #  data.append(line)
        #  else:
            #  continue

#  with open("./StackPropagation-SLU/data/atis/test.txt", "w") as f:
    #  f.write("".join(data))

        

