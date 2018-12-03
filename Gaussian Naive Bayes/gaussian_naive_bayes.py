"""
@author: DON
"""
import math

import pandas


class GausssianNB:

    def __init__(self, priors=None):
        self.__mean_variance = {}
        self.__priors = pandas.Series([priors])

    def fit(self, x_train: pandas.DataFrame, y_train: pandas.Series):
        self.__priors = y_train.value_counts()
        self.__priors /= self.__priors.sum()

        for _class, x_subset in x_train.groupby(y_train):
            mean = x_subset.mean()
            variance = x_subset.var()
            self.__mean_variance[_class] = (mean, variance)

    def predict(self, x_test: pandas.DataFrame) -> pandas.Series:
        return pandas.Series([
            self.__get_prediction(instance) for index, instance in x_test.iterrows()
        ])

    def __get_prediction(self, test_instance: pandas.Series):
        posterior_probabilities = {}
        for _class, prior_probability in self.__priors.items():
            likelihood = 0
            for attribute, value in test_instance.items():
                prob = self.__calculate_probability(
                    value,
                    self.__mean_variance[_class][0][attribute],
                    self.__mean_variance[_class][1][attribute]
                )
                if prob > 0:
                    likelihood += math.log(prob)
            posterior_probabilities[_class] = math.log(prior_probability) + likelihood

        return max(posterior_probabilities, key=lambda k: posterior_probabilities[k])

    @staticmethod
    def __calculate_probability(value: float, mean: float, variance: float) -> float:
        exponent = math.exp(-(math.pow(value - mean, 2) / (2 * variance)))
        return (1 / (math.sqrt(2 * math.pi * variance))) * exponent

    @staticmethod
    def accuracy_score(y_true: pandas.Series, y_pred: pandas.Series) -> float:
        accuracy = [test == pred for test, pred in zip(y_true, y_pred)]
        return sum(accuracy) / len(accuracy)


def main():
    diabetes_data_set = pandas.read_csv('diabetes_data.csv')

    train = diabetes_data_set.sample(frac=0.7)
    test = diabetes_data_set.drop(train.index)
    x_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    x_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    gaussian_nb = GausssianNB()
    gaussian_nb.fit(x_train, y_train)
    y_pred = gaussian_nb.predict(x_test)
    accuracy = gaussian_nb.accuracy_score(y_test, y_pred)
    print(f'Accuracy = {round(accuracy * 100, 2)}%')


if __name__ == '__main__':
    main()
