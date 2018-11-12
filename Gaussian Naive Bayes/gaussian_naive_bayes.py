"""
@author: DON
"""
import math
import pandas


class GausssianNB:

    def __init__(self, priors=None):
        self.__mean_variance = {}
        self.__priors = {} if priors is None else priors

    def __calculate_mean_variance(self, x_train, y_train):
        for _class in y_train.unique():
            filtered_class = x_train[(y_train == _class)]

            m_v = []
            for i in range(x_train.shape[1]):
                mean = filtered_class[i].mean()
                variance = math.pow(filtered_class[i].std(), 2)
                m_v.append((mean, variance))

            self.__mean_variance[_class] = m_v

    def fit(self, x_train, y_train):
        x_train = x_train.apply(pandas.to_numeric)
        y_train = y_train.apply(pandas.to_numeric)

        counts = y_train.value_counts().to_dict()
        self.__priors = {_class: count / y_train.size for _class, count in counts.items()}

        self.__calculate_mean_variance(x_train, y_train)

    @staticmethod
    def __calculate_probability(value, mean, variance):
        exponent = math.exp(-(math.pow(value - mean, 2) / (2 * variance)))
        return (1 / (math.sqrt(2 * math.pi * variance))) * exponent

    def __get_prediction(self, test_instance):
        posterior_probabilities = {}
        for _class, prior_probability in self.__priors.items():
            likelihood = 0
            for i in range(test_instance.size):
                prob = self.__calculate_probability(
                    test_instance[i],
                    self.__mean_variance[_class][i][0],
                    self.__mean_variance[_class][i][1])
                if prob > 0:
                    likelihood += math.log(prob)
            posterior_probabilities[_class] = math.log(prior_probability) + likelihood

        return max(posterior_probabilities, key=lambda k: posterior_probabilities[k])

    def predict(self, x_test):
        y_pred = pandas.Series([])
        for index, test_instance in x_test.iterrows():
            best_fit_class = self.__get_prediction(test_instance)
            y_pred = y_pred.append(pandas.Series([best_fit_class]))
        return y_pred

    @staticmethod
    def accuracy_score(y_true, y_pred):
        accuracy = 0
        for test, pred in zip(y_true, y_pred):
            if test == pred:
                accuracy += 1
        return accuracy / y_pred.size


def main():
    diabetes_data_set = pandas.read_csv('diabetes_data.csv', header=None)

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
