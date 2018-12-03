"""
@author: DON
"""

import pandas


class FindS:

    def __init__(self):
        self.hypothesis = None

    def fit(self, x_train: pandas.DataFrame, y_train: pandas.Series) -> pandas.Series:
        self.hypothesis = pandas.Series([None] * x_train.shape[1])

        for (index, row), target in zip(x_train.iterrows(), y_train):
            if target == 0:
                continue

            for i in range(len(self.hypothesis)):
                if self.hypothesis.iloc[i] is None:
                    self.hypothesis.iloc[i] = row.iloc[i]
                elif self.hypothesis.iloc[i] != row.iloc[i]:
                    self.hypothesis.iloc[i] = '?'

        return self.hypothesis


def main():
    play_data_set = pandas.read_csv('play_data.csv', header=None)
    x_train, y_train = play_data_set.iloc[:, :-1], play_data_set.iloc[:, -1]

    find_s = FindS()
    hypothesis = find_s.fit(x_train, y_train)
    print(f'Hypothesis = <{ hypothesis.to_frame().T.to_string(header=False, index=False) }>')


if __name__ == "__main__":
    main()
