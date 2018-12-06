"""
@author: DON
"""

import pandas
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def main():
    iris_data_set = pandas.read_csv('iris_data.csv')
    x_train, y_train = iris_data_set.iloc[:, :-1], iris_data_set.iloc[:, -1]

    k_means = KMeans(n_clusters=3)
    k_means.fit(x_train)
    k_means_y_pred = k_means.predict(x_train)
    k_means_accuracy = metrics.adjusted_rand_score(y_train, k_means_y_pred)
    print(f'K-means accuracy = {round(k_means_accuracy * 100, 2)}%')
    print('K-means confusion matrix: ')
    print(metrics.confusion_matrix(y_train, k_means_y_pred))
    print()

    gaussian_mixture = GaussianMixture(n_components=3)
    gaussian_mixture.fit(x_train)
    gaussian_mixture_y_pred = gaussian_mixture.predict(x_train)
    gaussian_mixture_accuracy = metrics.adjusted_rand_score(y_train, gaussian_mixture_y_pred)
    print(f'Gaussian Mixture accuracy = {round(gaussian_mixture_accuracy * 100, 2)}%')
    print('Gaussian Mixture confusion matrix: ')
    print(metrics.confusion_matrix(y_train, gaussian_mixture_y_pred))
    print()

    if k_means_accuracy > gaussian_mixture_accuracy:
        print('K-means is better than Gaussian mixture')
    elif k_means_accuracy < gaussian_mixture_accuracy:
        print('Gaussian mixture is better than K-means')
    else:
        print('Both K-means and Gaussian mixture are equally good')


if __name__ == "__main__":
    main()
