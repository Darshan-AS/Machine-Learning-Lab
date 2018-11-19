"""
@author: DON
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


def main():
    newsgroup_train_data_bunch = fetch_20newsgroups(subset='train')
    x_train, y_train = newsgroup_train_data_bunch.get('data'), newsgroup_train_data_bunch.get('target')

    newsgroup_test_data_bunch = fetch_20newsgroups(subset='test')
    x_test, y_test = newsgroup_test_data_bunch.get('data'), newsgroup_test_data_bunch.get('target')

    target_names = newsgroup_train_data_bunch.get('target_names')

    tfidf_multinomial_nb = make_pipeline(TfidfVectorizer(), MultinomialNB())
    tfidf_multinomial_nb.fit(x_train, y_train)
    y_pred = tfidf_multinomial_nb.predict(x_test)

    report = classification_report(y_test, y_pred, target_names=target_names)
    accuracy = accuracy_score(y_test, y_pred)
    print(report)
    print(f'Accuracy = {round(accuracy * 100, 2)}%')


if __name__ == '__main__':
    main()
