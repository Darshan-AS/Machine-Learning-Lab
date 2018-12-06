"""
@author: DON
"""

import pandas


class FindS:

    def __init__(self):
        self.hypothesis = None

    def fit(self, x_train: pandas.DataFrame, y_train: pandas.Series) -> pandas.Series:
        self.hypothesis = pandas.Series([None] * x_train.shape[1], index=x_train.columns)

        for (index, row), target in zip(x_train.iterrows(), y_train):
            if target == 0:
                continue

            for attribute, value in self.hypothesis.iteritems():
                if value is None:
                    self.hypothesis[attribute] = row[attribute]
                elif value != row[attribute]:
                    self.hypothesis[attribute] = '?'

        return self.hypothesis


def main():
    play_data_set = pandas.read_csv('enjoy_sport_data.csv')
    x_train, y_train = play_data_set.iloc[:, :-1], play_data_set.iloc[:, -1]

    find_s = FindS()
    hypothesis = find_s.fit(x_train, y_train)
    print(f'Hypothesis = <{ hypothesis.to_frame().T.to_string(header=False, index=False) }>')


if __name__ == "__main__":
    main()
