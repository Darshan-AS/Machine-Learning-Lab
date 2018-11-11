import pandas
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def main():
    play_data_set = pandas.read_csv('iris_data.csv', header=None)
    x_train, y_train = play_data_set.iloc[:, :-1], play_data_set.iloc[:, -1]

    k_means = KMeans(n_clusters=3)
    k_means.fit(x_train)
    k_means_accuracy = metrics.adjusted_rand_score(y_train, k_means.predict(x_train))
    print(f'K-means accuracy = {round(k_means_accuracy * 100, 2)}%')

    gaussian_mixture = GaussianMixture(n_components=3)
    gaussian_mixture.fit(x_train)
    gaussian_mixture_accuracy = metrics.adjusted_rand_score(y_train, gaussian_mixture.predict(x_train))
    print(f'Gaussian Mixture accuracy = {round(gaussian_mixture_accuracy * 100, 2)}%')

    if k_means_accuracy > gaussian_mixture_accuracy:
        print('K-means is better than Gaussian mixture')
    elif k_means_accuracy < gaussian_mixture_accuracy:
        print('Gaussian mixture is better than K-means')
    else:
        print('Both K-means and Gaussian mixture are equally good')


if __name__ == "__main__":
    main()
