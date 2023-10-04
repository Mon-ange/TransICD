
class TopEvaluator(object):

    def __init__(self, top_number):
        self.top_number = top_number

    def evaluate(self, predicts, targets):
        """
            if one of the top_number of one predict hit the correspondent target, we calculate it as a true case
            predicts shape: [ [0.25, 0.36, 0.01 ... 0.02], [0.33, 0.3, 0.1 ...] ...] N x L, N is the number of cases, L is the number of Labels
        """
        true_case_number = 0
        total_case_number = 0

        return float(true_case_number) / float(total_case_number)
