from fscore import FScore

from absl import flags
flags.D

class IncrementalEval(object):

    """Create an incremental evaluation metric."""

    def __init__(self, model, an):
        """TODO: to be defined.

        :model: TODO
        :an: TODO

        """
        self._model = model
        self._an = an
        
