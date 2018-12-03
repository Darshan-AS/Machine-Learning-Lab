"""
@author: DON
"""

import math

import pandas


class DecisionTreeID3:
    class Node:

        def __init__(self):
            self.label = None
            self.children = {}

        def __str__(self):
            string = '[' + self.label
            for value, node in self.children.items():
                string += f' [{value}: {str(node)}], '
            string += ']'
            return string

    def __init__(self):
        self.root = None

    def fit(self, x_train: pandas.DataFrame, y_train: pandas.Series):
        self.root = self.__id3(x_train, y_train)

    def predict(self, x_test: pandas.DataFrame):
        y_pred = [self.__classify(row, self.root) for index, row in x_test.iterrows()]
        return pandas.Series(y_pred)

    def __id3(self, x: pandas.DataFrame, y: pandas.Series) -> Node:
        node = self.Node()

        if y.nunique() == 1:
            node.label = y.iloc[0]
            return node

        if x.empty:
            node.label = y.value_counts().idxmax()
            return node

        node.label = self.__get_best_attribute(x, y)
        for value, x_subset in x.groupby(node.label):
            y_subset = y[x_subset.index]
            node.children[value] = self.__id3(x_subset, y_subset)

        return node

    def __get_best_attribute(self, x: pandas.DataFrame, y: pandas.Series) -> str:
        x_entropy = self.__get_entropy(y)
        information_gains = pandas.Series([])
        for attribute, series in x.iteritems():
            attribute_entropy = [(y[series == value].size / y.size) * self.__get_entropy(y[series == value])
                                 for value in series.unique()]
            information_gains[attribute] = x_entropy - sum(attribute_entropy)
        return information_gains.idxmax()

    @staticmethod
    def __get_entropy(y: pandas.Series) -> float:
        probs = y.value_counts()
        probs /= probs.sum()
        return sum([-prob * math.log(prob, 2) for prob in probs])

    def __classify(self, instance: pandas.Series, tree: Node) -> str:
        if not tree.children:
            return tree.label
        return self.__classify(instance, tree.children[instance[tree.label]])


def main():
    play_tennis_data_set = pandas.read_csv('play_tennis_data.csv')
    x_train, y_train = play_tennis_data_set.iloc[:, :-1], play_tennis_data_set.iloc[:, -1]

    decision_tree_id3 = DecisionTreeID3()
    decision_tree_id3.fit(x_train, y_train)

    x_test = pandas.DataFrame()
    t1 = pandas.Series(['sunny', 'cool', 'normal', 'weak'], index=x_train.columns)
    t2 = pandas.Series(['rainy', 'hot', 'normal', 'strong'], index=x_train.columns)
    # Add more test cases here
    x_test = x_test.append([t1, t2], ignore_index=True)
    y_pred = decision_tree_id3.predict(x_test)
    print(y_pred)


if __name__ == '__main__':
    main()
