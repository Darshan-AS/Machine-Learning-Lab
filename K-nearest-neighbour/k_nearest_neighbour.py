"""
@author: DON
"""
import pandas
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    iris_data_bunch = load_iris()
    iris_data = iris_data_bunch.get('data')
    iris_target = iris_data_bunch.get('target')
    iris_target_names = iris_data_bunch.get('target_names')
    iris_features = iris_data_bunch.get('feature_names')

    x_train, x_test, y_train, y_test = train_test_split(
        iris_data, iris_target, test_size=0.3, shuffle=True, random_state=77)

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(x_train, y_train)
    y_pred = knn_classifier.predict(x_test)

    print_test_results(x_test, y_test, y_pred, iris_features, iris_target_names)


def print_test_results(x_test, y_test, y_pred, features, feature_names):
    result = pandas.DataFrame(x_test, columns=features)
    result['Actual Target'] = [feature_names[i] for i in y_test]
    result['Predicted target'] = [feature_names[i] for i in y_pred]
    print(result)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f'Accuracy = {round(accuracy * 100, 2)}%')


if __name__ == '__main__':
    main()
