"""
@author: DON
"""
import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas


def main():
    iris_data_bunch = load_iris()
    print(iris_data_bunch)
    iris_data = iris_data_bunch.get('data')
    iris_target = iris_data_bunch.get('target')
    iris_target_names = iris_data_bunch.get('target_names')
    iris_features = iris_data_bunch.get('feature_names')

    x_train, x_test, y_train, y_test = train_test_split(
        iris_data, iris_target, test_size=0.3, shuffle=True, random_state=77)

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(x_train, y_train)
    y_predicted = knn_classifier.predict(x_test)

    print_test_results(x_test, y_test, y_predicted, iris_features, iris_target_names)


def print_test_results(x_test, y_test, y_predicted, features, feature_names):
    result = pandas.DataFrame(x_test, columns=features)
    result['Actual Target'] = numpy.apply_along_axis(
        lambda y: [feature_names[i] for i in y_test], 0, y_test)
    result['Predicted target'] = numpy.apply_along_axis(
        lambda y: [feature_names[i] for i in y_predicted], 0, y_predicted)
    print(result)
    print('Accuracy =', metrics.accuracy_score(y_test, y_predicted))


if __name__ == '__main__':
    main()
