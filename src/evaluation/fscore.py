from tabulate import tabulate

class FScore(object):

    """FScore."""

    def __init__(self,correct=0, predcount=0, goldcount=0):
        """Initiate the F-score instance

        :correct: TODO
        :predcount: TODO
        :goldcount: TODO

        """
        self._correct = correct
        self._predcount = predcount
        self._goldcount = goldcount
        self._str_format ="{:>8}{:>8}{:>8}{:>8.2f}{:>8.2f}{:>8.2f}"

    def precision(self):
        if self._predcount > 0:
            return (100.0 * self._correct) / self._predcount
        else:
            return 0.0

    def recall(self):
        if self._goldcount > 0:
            return (100.0 * self._correct) / self._goldcount
        else:
            return 0.0

    def fscore(self):
        precision = self.precision()
        recall = self.recall()
        if (precision + recall) > 0:
            return (2 * precision * recall) / (precision + recall)
        else:
            return 0.0

    def __str__(self):
        precision = self.precision()
        recall = self.recall()
        fscore = self.fscore()
        str_header = ["#Pred", "#Gold", "#Correct", "Precision", "Recall", "F-score"]
        output = [[self._predcount, self._goldcount, self._correct, precision, recall, fscore]]
        return tabulate(output, headers=str_header, tablefmt="github", floatfmt=".3f")

    def __iadd__(self, other):
        self._correct += other._correct
        self._predcount += other._predcount
        self._goldcount += other._goldcount
        return self

    def __add__(self, other):
        return FScore(self._correct + other._correct, self._predcount + other._predcount, self._goldcount + other._goldcount)

