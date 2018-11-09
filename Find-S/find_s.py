"""
@author: DON
"""

import pandas


class FindS:

    def __init__(self):
        self.hypothesis = None
        pass

    def fit(self, x_train, y_train):
        self.hypothesis = pandas.Series([None] * len(x_train.columns))

        for (index, row), target in zip(x_train.iterrows(), y_train):
            if target == 0:
                continue

            for i in range(len(self.hypothesis)):
                if self.hypothesis.iloc[i] == row.iloc[i]:
                    continue
                elif self.hypothesis.iloc[i] is None:
                    self.hypothesis.iloc[i] = row.iloc[i]
                else:
                    self.hypothesis.iloc[i] = '?'

        return self.hypothesis

    def predict(self, x_test):
        y_pred = pandas.Series([])
        index = self.hypothesis != '?'

        for i, row in x_test.iterrows():

            if self.hypothesis[index].equals(row[index]):
                y_pred = y_pred.append(pandas.Series([1]))
            else:
                y_pred = y_pred.append(pandas.Series([0]))

        return y_pred

    @staticmethod
    def score(y_test, y_pred):
        accuracy = 0
        for test, pred in zip(y_test, y_pred):
            if test == pred:
                accuracy += 1

        return accuracy / y_pred.size


def main():
    play_data_set = pandas.read_csv('play_data.csv', header=None)

    train = play_data_set.sample(frac=0.6)
    test = play_data_set.drop(train.index)
    x_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    x_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    find_s = FindS()
    hypothesis = find_s.fit(x_train, y_train)
    print(f'Hypothesis = { hypothesis.to_frame().T.to_string(header=False, index=False) }')

    y_pred = find_s.predict(x_test)
    accuracy = find_s.score(y_test, y_pred)
    print('Accuracy = {}%'.format(round(accuracy * 100, 2)))


if __name__ == "__main__":
    main()
