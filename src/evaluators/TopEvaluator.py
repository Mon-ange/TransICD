import numpy as np


class TopEvaluator(object):

    def __init__(self, top_number):
        self.top_number = top_number

    def evaluate(self, predicts, targets):
        """
            if one of the top_number of one predict hit the correspondent target, we calculate it as a true case
            predicts shape: [ [0.25, 0.36, 0.01 ... 0.02], [0.33, 0.3, 0.1 ...] ...] N x L, N is the number of cases, L is the number of Labels
        """
        print(predicts[0])
        total_case_number = np.array(predicts).shape[0]
        L = np.array(predicts).shape[1]
        count = 0
        for i in range(len(predicts)):
            true_tar_no = 0
            for k in range(len(targets[i])):  # select the true
                if targets[i][k] == 1:
                    true_tar_no = k

            flag = 0
            array = np.array(predicts[i])
            d = dict(enumerate(array.flatten(),0))

            sorted_d = sorted(d.items(),key=lambda x:x[1],reverse=True)

            resarray = [0] *self.top_number

            for l in range(len(sorted_d)):
                resarray[l]=sorted_d[l][0]
                l+=1
                if l== self.top_number:
                    break
            if i ==0:
                print(resarray)

            for result in resarray:
                if result == true_tar_no:
                    flag = 1

            if flag == 1:
                count += 1

        true_case_number = count
        # total_case_number = 0

        return float(true_case_number) / float(total_case_number)


if __name__ == '__main__':
    topEvaluator = TopEvaluator(3)
    predicts = [[0.9, 0.4, 0.8, 0.3],
                [0.4, 0.5, 0.9, 0.1],
                [0.5,0.5,0.1,0.3],
                [0.1,0.3,0.4,0.2],
                [0.1,0.4,0,0]]
    targets = [[0, 0, 1, 0],
               [0,0,0,1],
               [1,0,0,0],
               [1,0,0,0],
               [1,0,0,0]]
    print(TopEvaluator.evaluate(topEvaluator,predicts, targets))
