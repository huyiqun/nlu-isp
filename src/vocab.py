from collections import OrderedDict
from ordered_set import OrderedSet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.logging_util import ColoredLog
VERBOSE=10
logger = ColoredLog(__name__, verbose=VERBOSE)
logger.info("abc")
from transformers import BertTokenizer
from transformers import PreTrainedTokenizer
pt = PreTrainedTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Here is the sentence for embedding."
tokenizer.tokenize(text)
a = iter(tokenizer.vocab.keys())
tokenizer.tokenize('wordpiece')
tokenizer.convert_tokens_to_ids('wordpiece')
tokenizer.convert_ids_to_tokens(100)
tokenizer.convert_tokens_to_ids(['me', 'embedding'])
tokenizer.encode("this is work")
print(next(a))
tokenizer.encode('embedding', add_special_tokens=True)
tokenizer.convert_ids_to_tokens(102)

train =[]
train_full = []
with open('./data/atis/train.txt', 'r') as f:
    dp = {}
    wait_for_id = True
    dp['tokens'] = []
    dp['tags'] = []
    for line in f:
        l = line.split()
        if len(l) == 0:
            dp = {}
            wait_for_id = True
            dp['tokens'] = []
            dp['tags'] = []
        elif len(l) == 1:
            if wait_for_id:
                dp['id'] = l[0]
                wait_for_id = False
            else:
                dp['intent'] = l[0]
                if dp['id'].endswith('full'):
                    train_full.append(dp)
                train.append(dp)
        else:
            dp['tokens'].append(l[0])
            dp['tags'].append(l[1])

"\n".split()

torch.manual_seed(1)
for i in range(10):
    print(tokenizer.convert_tokens_to_ids(train_full[i]['tokens']))

word_to_idx = {'hello': 1, 'world': 1}
embeds = nn.Embedding(2, 5)
lookup_tensor = torch.tensor([word_to_idx['hello']], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
torch.tensor()

class Vocabulary(object):

    """Abstract vocabulary structure for word tokens, intent, and slots."""

    def __init__(self):
        """TODO: to be defined. """
        self.counter = Counter()

    def index2
        

class Vocabulary(object):
    """
    Storage and serialization a set of elements.
    """

    def __init__(self, name, if_use_pad, if_use_unk):

        self.__name = name
        self.__if_use_pad = if_use_pad
        self.__if_use_unk = if_use_unk

        self.__index2instance = OrderedSet()
        self.__instance2index = OrderedDict()

        # Counter Object record the frequency
        # of element occurs in raw text.
        self.__counter = Counter()

        if if_use_pad:
            self.__sign_pad = "<PAD>"
            self.add_instance(self.__sign_pad)
        if if_use_unk:
            self.__sign_unk = "<UNK>"
            self.add_instance(self.__sign_unk)

    @property
    def name(self):
        return self.__name

    def add_instance(self, instance):
        """ Add instances to alphabet.

        1, We support any iterative data structure which
        contains elements of str type.

        2, We will count added instances that will influence
        the serialization of unknown instance.

        :param instance: is given instance or a list of it.
        """

        if isinstance(instance, (list, tuple)):
            for element in instance:
                self.add_instance(element)
            return

        # We only support elements of str type.
        assert isinstance(instance, str)

        # count the frequency of instances.
        self.__counter[instance] += 1

        if instance not in self.__index2instance:
            self.__instance2index[instance] = len(self.__index2instance)
            self.__index2instance.append(instance)

    def get_index(self, instance):
        """ Serialize given instance and return.

        For unknown words, the return index of alphabet
        depends on variable self.__use_unk:

            1, If True, then return the index of "<UNK>";
            2, If False, then return the index of the
            element that hold max frequency in training data.

        :param instance: is given instance or a list of it.
        :return: is the serialization of query instance.
        """

        if isinstance(instance, (list, tuple)):
            return [self.get_index(elem) for elem in instance]

        assert isinstance(instance, str)

        try:
            return self.__instance2index[instance]
        except KeyError:
            if self.__if_use_unk:
                return self.__instance2index[self.__sign_unk]
            else:
                max_freq_item = self.__counter.most_common(1)[0][0]
                return self.__instance2index[max_freq_item]

    def get_instance(self, index):
        """ Get corresponding instance of query index.

        if index is invalid, then throws exception.

        :param index: is query index, possibly iterable.
        :return: is corresponding instance.
        """

        if isinstance(index, list):
            return [self.get_instance(elem) for elem in index]

        return self.__index2instance[index]

    def save_content(self, dir_path):
        """ Save the content of alphabet to files.

        There are two kinds of saved files:
            1, The first is a list file, elements are
            sorted by the frequency of occurrence.

            2, The second is a dictionary file, elements
            are sorted by it serialized index.

        :param dir_path: is the directory path to save object.
        """

        # Check if dir_path exists.
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        list_path = os.path.join(dir_path, self.__name + "_list.txt")
        with open(list_path, 'w') as fw:
            for element, frequency in self.__counter.most_common():
                fw.write(element + '\t' + str(frequency) + '\n')

        dict_path = os.path.join(dir_path, self.__name + "_dict.txt")
        with open(dict_path, 'w') as fw:
            for index, element in enumerate(self.__index2instance):
                fw.write(element + '\t' + str(index) + '\n')

    def __len__(self):
        return len(self.__index2instance)

    def __str__(self):
        return 'Alphabet {} contains about {} words: \n\t{}'.format(self.name, len(self), self.__index2instance)
