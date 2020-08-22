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
import time
import random
import operator
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Process(object):
    """ A Process object handles the calculation of the Model object, including the process of training (loss function, optimizer), validation and inference. """

    def __init__(self, model, proc_args):
        """ Initialization.

        :model: model architecture (type: nn.Module)
        :proc_args: additional process parameters
        """
        self._model = model
        self._proc_args = proc_args

        if torch.cuda.is_available():
            time_start = time.time()
            self._model = self._model.cuda()
            print(f"The model has been loaded into GPU and cost {time.time() - time_start} seconds.\n")

        self._criterion = nn.NLLLoss()
        self._optimizer = optim.Adam(
            self._model.parameters(), lr=self._proc_args.learning_rate,
            weight_decay=self._proc_args.l2_penalty
        )

    def train(self, dataset):
        """ Train the model.

        :dataset: the training set
        """

        best_dev_intent = 0.0
        best_dev_sent = 0.0

        dataloader = dataset.batch_delivery(self._proc_args.batch_size)
        for epoch in range(0, self._proc_args.num_epoch):
            total_slot_loss, total_intent_loss = 0.0, 0.0

            time_start = time.time()
            self._model.train()

            for text_batch, slot_batch, intent_batch, anticipation_batch, _ in tqdm(dataloader, ncols=50):
                padded_text, [sorted_slot, sorted_intent, sorted_anticipation], seq_lens, _ = dataset.add_padding(
                    text_batch, [(slot_batch, False), (intent_batch, False), (anticipation_batch, False)]
                )
                sorted_intent = [item * num for item, num in zip(sorted_intent, seq_lens)]
                sorted_intent = list(Process.expand_list(sorted_intent))

                text_var = Variable(torch.LongTensor(padded_text))
                slot_true = Variable(torch.LongTensor(list(Process.expand_list(sorted_slot))))
                intent_true = Variable(torch.LongTensor(sorted_intent))
                anticipation_true = Variable(torch.LongTensor(sorted_anticipation))

                if torch.cuda.is_available():
                    text_var = text_var.cuda()
                    slot_var = slot_var.cuda()
                    intent_var = intent_var.cuda()

                # teacher forcing
                random_slot, random_intent = random.random(), random.random()
                if random_slot < self._proc_args.slot_forcing_rate and \
                        random_intent < self._proc_args.intent_forcing_rate:
                    slot_out, intent_out, anticipation_out = self._model(
                        text_var, seq_lens, forced_slot=slot_var, forced_intent=intent_var
                    )
                elif random_slot < self._proc_args.slot_forcing_rate:
                    slot_out, intent_out, anticipation_out = self._model(
                        text_var, seq_lens, forced_slot=slot_var
                    )
                elif random_intent < self._proc_args.intent_forcing_rate:
                    slot_out, intent_out, anticipation_out = self._model(
                        text_var, seq_lens, forced_intent=intent_var
                    )
                else:
                    slot_out, intent_out, anticipation_out = self._model(text_var, seq_lens)

                # loss calculation
                slot_loss = self._criterion(slot_out, slot_true)
                intent_loss = self._criterion(intent_out, intent_true)
                anticipation_loss = self._criterion(anticipation_out, anticipation_true)
                batch_loss = slot_loss + intent_loss + self._proc_args.lmbda * anticipation_loss

                self._optimizer.zero_grad()
                batch_loss.backward()
                self._optimizer.step()

                try:
                    total_slot_loss += slot_loss.cpu().item()
                    total_intent_loss += intent_loss.cpu().item()
                except AttributeError:
                    total_slot_loss += slot_loss.cpu().data.numpy()[0]
                    total_intent_loss += intent_loss.cpu().data.numpy()[0]

            time_con = time.time() - time_start
            print(f"[Epoch {epoch}]: The total slot loss on train data is {total_slot_loss:2.6f}, intent data is {total_intent_loss:2.6f}, cost about {time_con:2.6f} seconds.")

            change, time_start = False, time.time()

            # evaluate on the dev set, if performance is better, save the model
            pred_slot, real_slot, pred_intent, real_intent, _, _, _ = Process.predict(self._model, dataset, self._proc_args.batch_size)
            dev_acc = Process.accuracy(pred_intent, real_intent)
            dev_sent_acc = Process.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)

            if dev_acc > best_dev_intent or dev_sent_acc > best_dev_sent:
                if dev_acc > best_dev_intent:
                    best_dev_intent = dev_acc
                if dev_sent_acc > best_dev_sent:
                    best_dev_sent = dev_sent_acc

                model_save_dir = os.path.join(self._proc_args.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)

                torch.save(self._model, os.path.join(model_save_dir, "model.pkl"))

                time_con = time.time() - time_start
                print('[Epoch {:2d}]: In validation process, the intent acc is {:2.6f}, ' \
                      'the semantic acc is {:.2f}, cost about {:2.6f} seconds.\n'.format(epoch, dev_acc, dev_sent_acc, time_con))

    @staticmethod
    def predict(model, dataset, batch_size):
        """ Generate prediction.

        :model: a pytorch model
        :dataset: a pytorch Dataset
        :batch_size: batch size
        """
        model.eval()

        dataloader = dataset.batch_delivery(batch_size=batch_size)

        pred_slot, real_slot = [], []
        pred_intent, real_intent = [], []
        sorted_intents = []
        sorted_ids = []
        text = []

        for text_batch, slot_batch, intent_batch, ids_batch in tqdm(dataloader, ncols=50):
            padded_text, [sorted_slot, sorted_intent], seq_lens, sorted_index = dataset.add_padding(
                text_batch, [(slot_batch, False), (intent_batch, False)], digital=False
            )
            sorted_intents.extend(sorted_intent)
            text.extend(padded_text)
            sorted_ids.extend(list(np.array(ids_batch)[sorted_index]))

            real_slot.extend(sorted_slot)
            real_intent.extend(list(Process.expand_list(sorted_intent)))

            digit_text = dataset.word_alphabet.get_index(padded_text)
            var_text = Variable(torch.LongTensor(digit_text))

            if torch.cuda.is_available():
                var_text = var_text.cuda()

            slot_idx, intent_idx = model(var_text, seq_lens, n_predicts=1)
            nested_slot = Process.nested_list([list(Process.expand_list(slot_idx))], seq_lens)[0]
            pred_slot.extend(dataset.slot_alphabet.get_instance(nested_slot))
            nested_intent = Process.nested_list([list(Process.expand_list(intent_idx))], seq_lens)[0]
            pred_intent.extend(dataset.intent_alphabet.get_instance(nested_intent))

        exp_pred_intent = Process.max_freq_predict(pred_intent)
        return pred_slot, real_slot, exp_pred_intent, real_intent, pred_intent, text, sorted_ids

    @staticmethod
    def save_test(model_path, dataset_path, batch_size):
        """ Save the prediction results for the test set for further evaluation.

        :model_path: path to the pytorch model
        :dataset_path: path to the dataset
        :batch_size: batch size to process test set
        """

        model = torch.load(model_path)
        dataset = torch.load(dataset_path)

        # generate prediction
        pred_slot, real_slot, exp_pred_intent, real_intent, pred_intent, text, sorted_ids = Process.predict(
            model, dataset, "test", batch_size
        )

        pred = {}
        pred["pred_slot"] = pred_slot
        pred["golden_slot"] = real_slot
        pred["golden"] = real_intent
        pred["pred"] = exp_pred_intent
        pred["token_level"] = pred_intent
        pred["text"] = text
        pred["sorted_ids"] = sorted_ids

        pred_res_dir = os.path.join(dataset.save_dir, "results")
        if not os.path.exists(pred_res_dir):
            os.mkdir(pred_res_dir)
        torch.save(pred, os.path.join(pred_res_dir, "test.pkl"))

        intent_acc = Process.accuracy(exp_pred_intent, real_intent)
        sent_acc = Process.semantic_acc(pred_slot, real_slot, exp_pred_intent, real_intent)

        return (intent_acc, sent_acc), (pred_slot, real_slot, exp_pred_intent, real_intent, pred_intent, text, sorted_ids)

    @staticmethod
    def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
        """ Compute full query semantic accuracy.

        :pred_slot: slot prediction
        :real_slot: golden slot labels
        :pred_intent: intent prediction
        :real_intent: golden intent labels
        """

        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):

            if p_slot == r_slot and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def accuracy(pred_list, real_list):
        """ Calculate accuracy if given two vectors

        :pred_list: the predicted values
        :real_list: the true labels
        """

        pred_array = np.array(list(Process.expand_list(pred_list)))
        real_array = np.array(list(Process.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def f1_score(pred_list, real_list):
        """ Calculate f1 score if given two vectors

        :pred_list: the predicted values
        :real_list: the true labels
        """

        tp, fp, fn = 0.0, 0.0, 0.0
        for i in range(len(pred_list)):
            seg = set()
            result = [elem.strip() for elem in pred_list[i]]
            target = [elem.strip() for elem in real_list[i]]

            j = 0
            while j < len(target):
                cur = target[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(target):
                        str_ = target[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    seg.add((cur, j, k - 1))
                    j = k - 1
                j = j + 1

            tp_ = 0
            j = 0
            while j < len(result):
                cur = result[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(result):
                        str_ = result[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    if (cur, j, k - 1) in seg:
                        tp_ += 1
                    else:
                        fp += 1
                    j = k - 1
                j = j + 1

            fn += len(seg) - tp_
            tp += tp_

        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        return 2 * p * r / (p + r) if p + r != 0 else 0

    @staticmethod
    def max_freq_predict(sample, weights=None):
        """ Get sentence-level intent prediction by majority voting

        :sample: votes from tokens
        :weights: weight each token gets in determining sentence-level intent
        """
        predict = []
        for vote, weight in zip(sample, weights):
            res = defaultdict(float)
            for v, w in zip(vote, weight):
                res[v] += weight
            predict.append(max(res.iteritems(), key=operator.itemgetter(1))[-1])
        return predict

    @staticmethod
    def expand_list(nested_list):
        """ Expand nested list. """

        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Process.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        """ Segment a long list into nested list with size `seq_lens`. """

        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items
