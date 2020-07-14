import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

res.keys()
summary_intent = defaultdict(list)



for k in res:
    l = len(res[k])

    summary_intent[k].append()



def convert_to_ml(intent):
    vec = np.zeros(len(label_set), dtype=int)
    for i in intent.split("#"):
        vec[label_set[i]] = 1
    return list(vec)

label_set = {}
for i in real_intent:
    for j in i.split("#"):
        if j not in label_set:
            label_set[j] = len(label_set)


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
