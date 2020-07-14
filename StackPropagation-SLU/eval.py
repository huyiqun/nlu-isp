"""
@Author		:           Lee, Qin
@StartTime	:           2018/08/13
@Filename	:           train.py
@Software	:           Pycharm
@Framework  :           Pytorch
@LastModify	:           2019/05/07
"""

from utils.module import ModelManager
from utils.loader import DatasetManager
from utils.process import Processor
from utils.fscore import FScore
from utils.logging_utils import ColoredLog
import sys
import os
from collections import defaultdict, Counter
sys.path.append("./StackPropagation-SLU")

import torch

import os
import json
import random
import argparse
import numpy as np
#  from sklearn.metrics import f1_score, accuracy_score, precisirn_recall_fscore_support

parser = argparse.ArgumentParser()

# Training parameters.
parser.add_argument('--data_dir', '-dd', type=str, default='./StackPropagation-SLU/data/atis')
parser.add_argument('--save_dir', '-sd', type=str, default='./StackPropagation-SLU/data/atis/save')
#  parser.add_argument('--data_dir', '-dd', type=str, default='./data/atis')
#  parser.add_argument('--save_dir', '-sd', type=str, default='./data/atis/save')
parser.add_argument("--random_state", '-rs', type=int, default=0)
parser.add_argument('--num_epoch', '-ne', type=int, default=100)
parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.001)
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.4)
parser.add_argument('--intent_forcing_rate', '-ifr', type=float, default=0.9)
parser.add_argument("--differentiable", "-d", action="store_true", default=False)
parser.add_argument('--slot_forcing_rate', '-sfr', type=float, default=0.9)

# model parameters.
parser.add_argument('--word_embedding_dim', '-wed', type=int, default=64)
parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=256)
parser.add_argument('--intent_embedding_dim', '-ied', type=int, default=8)
parser.add_argument('--slot_embedding_dim', '-sed', type=int, default=32)
parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=64)
parser.add_argument('--intent_decoder_hidden_dim', '-idhd', type=int, default=64)
parser.add_argument('--attention_hidden_dim', '-ahd', type=int, default=1024)
parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)

if __name__ == "__main__":
    args = parser.parse_args()

    # Save training and model parameters.
    if not os.path.exists(args.save_dir):
        os.system("mkdir -p " + args.save_dir)

    log_path = os.path.join(args.save_dir, "param.json")
    with open(log_path, "w") as fw:
        fw.write(json.dumps(args.__dict__, indent=True))

    # Fix the random seed of package random.
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # Fix the random seed of Pytorch when using GPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
        torch.cuda.manual_seed(args.random_state)

    # Fix the random seed of Pytorch when using CPU.
    torch.manual_seed(args.random_state)
    torch.random.manual_seed(args.random_state)

    # Instantiate a dataset object.
    dataset = DatasetManager(args)
    dataset.quick_build()
    dataset.show_summary()
    #  dataset.ids["test"]

    # Instantiate a network model object.
    model = ModelManager(
        args, len(dataset.word_alphabet),
        len(dataset.slot_alphabet),
        len(dataset.intent_alphabet))
    model.show_summary()

    # To train and evaluate the models.
    process = Processor(dataset, model, args.batch_size)
    process.train()

    print('\nAccepted performance: ' + str(Processor.validate(
        os.path.join(args.save_dir, "model/model.pkl"),
        os.path.join(args.save_dir, "model/dataset.pkl"),
        args.batch_size)) + " at test dataset;\n")
    torch.save(dataset, os.path.join(args.save_dir, "model/dataset.pkl"))
    model = torch.load("./StackPropagation-SLU/data/atis/save/model/model.pkl")
    dataset = torch.load("./StackPropagation-SLU/data/atis/save/model/dataset.pkl")

    class IncTestOutcome(object):
    
        """Incremental Test Outcome Object."""
    
        def __init__(self, sent_id):
            """Initialization.
            """
            self._sent_id = sent_id
            self.items = defaultdict(dict)
            self.max_len = -1
            self.logger = ColoredLog(__name__)

        def update(self, golden, pred, golden_slot, pred_slot, text, length):
            if length != "full":
                length = int(length)
                self.items[length]["golden"] = golden
                self.items[length]["pred"] = pred
                self.items[length]["golden_slot"] = golden_slot
                self.items[length]["pred_slot"] = pred_slot
                self.items[length]["text"] = text[:length+1]
                self.max_len = max(self.max_len, length + 1)

        def pprint(self):
            out = []

            for i in range(1, self.max_len):
                #  j = str(i)
                j = i
                out.append([i, "-", self.items[j]["golden"], self.items[j]["pred"], "-", "-"])
                for k in range(1, i+1):
                    out.append(["-", self.items[j]["text"][k], "-", "-", self.items[j]["golden_slot"][k], self.items[j]["pred_slot"][k]])
                out.append(["==="] * 6)
            self.logger.critical(out, header=["len", "text", "golden", "pred", "golden_slot", "pred_slot"])


    pred_slot, real_slot, exp_pred_intent, real_intent, pred_intent, text, sorted_ids = process.prediction(model, dataset, "test", 200)

    test_set = {}
    #  test_set["test-00011"].pprint()

    for i in range(len(dataset.ids["test"])):
    #  for i in range(20):
        sent_id = sorted_ids[i][:10]
        sub_id = sorted_ids[i][11:]
        if sent_id not in test_set:
            test_set[sent_id] = IncTestOutcome(sent_id)
        
        #  if sub_id != "full":
            #  print(text[i][:int(sub_id)+1])
            #  print(golden_slot)
        test_set[sent_id].update(real_intent[i], exp_pred_intent[i], real_slot[i], pred_slot[i], text[i], sub_id)

    res = {s:test_set[s].items for s in test_set}
    res_dir = os.path.join(args.save_dir, "results")
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    torch.save(res, os.path.join(res_dir, "test.pkl"))

        #  if not sorted_ids[i].endswith("full"):
            #  length = int(sorted_ids[10:])
            #  test_set[sent_id].length.append(length)
            #  test_set[sent_id].update()

        #  test_dp[sent_id].append(pred_intent[i][1:])




        #  test_pred
        #  test_golden[sent_id].append(real_intent[i])
        #  test_text[sent_id].append(text[i])

    #  for k in test_dp:
        #  y_true = np.array(list(map(convert_to_ml, test_golden[k])))
        #  y_pred = np.array(list(map(convert_to_ml, list(map(lambda x: Counter(x).most_common()[0][0], test_dp[k])))))

    #  test_text["test-00001"]
    #  test_golden["test-00001"]
    #  test_dp["test-00001"]
    #  f1_score(y_true, y_pred, average="weighted")

    #  Counter(test_dp["test-00001"][0]).most_common()[0][0]
    #  list(map(lambda x: Counter(x).most_common()[0][0], test_dp["test-00001"]))

    #  f1_score(real_intent, exp_pred_intent, average="samples")
    #  accuracy_score(real_intent, exp_pred_intent)
    #  f1_score(np.array([[1,0],[1,0],[1,1]]), np.array([[0,1],[1,0],[1,1]]), average="samples")
    #  f1_score(np.array([[1,0],[1,1],[1,1]]), np.array([[0,1],[1,0],[1,1]]), average="samples")




    #  for test_id in test_dp.keys():
        #  ++  

    #  sorted_len = np.argsort(l)

    #  test_dp["test-00001"]
    #  len(test_golden["test-00001"])
    #  np.array(test_text["test-00001"])[sorted_len]
    #  for k in sorted_len:
        #  print(test_dp["test-00001"][k])
        #  print()
    #  [test_dp["test-00001"]]
    #  np.array(test_text["test-00001"])[sorted_len]
    #  test_dp.keys()
    #  print([len(i) for i in test_dp["test-00001"]])

    #  f1_score(ml_golden, ml_pred, average="weighted")
    #  np.array(list(ml_golden))
    #  print(len(sorted_ids))

    #  foo = dict()


    #  np.unique(real_intent)
    #  np.unique(exp_pred_intent)
    #  Counter(real_intent)
    #  dataset = torch.load("./StackPropagation-SLU/save/model/dataset.pkl")
    #  dataset.show_summary()
    #  dataset.get_dataset("train", False)
    #  model = torch.load("./StackPropagation-SLU/save/model/model.pkl")
    #  correct = len(np.where(np.array(real_intent)==np.array(exp_pred_intent))[0])
    #  FScore(correct, )

    #  process = Processor(dataset, model, 20)
    #  process.estimate(if_dev=False, test_batch=20)
    #  pred_slot, real_slot, exp_pred_intent, real_intent, pred_intent, text, sorted_ids = process.prediction(model, dataset, "test", 10)
    #  len(real_intent)
    #  dataset
    #  sorted_intents[0][197]
    #  sorted_intents[1][197]
    #  logger.info(np.hstack((np.arange(893)[:, np.newaxis], np.transpose(sorted_intents)[0])), header=["id", "batch", "sorted"])
    #  np.transpose(sorted_intents)[0].shape
    #  np.array(sorted_intents).shape
    #  len(sorted_intents)
    #  logger.info(np.hstack((np.arange(893)[:, np.newaxis], np.array(real_intent)[:, np.newaxis], np.array(exp_pred_intent)[:, np.newaxis]))[:20], header=["id", "golden", "pred"])
    #  for i in range(195, 199):
        #  print(text[i])
        #  print(real_intent[i])
        #  print(pred_intent[i])
        #  print(exp_pred_intent[i])
    #  pred_slot[197]
    #  real_slot[197]
    #  real_slot[0]
    #  exp_pred_intent[197]
    #  pred_intent[197]
    #  len(text)
    #  text[197]
    #  real_intent[19]
    #  for i in range(len(pred_intent)):
        #  if len(np.unique(pred_intent[i])) > 1:
            #  print(i)
            #  break
    #  len(exp_pred_intent)

