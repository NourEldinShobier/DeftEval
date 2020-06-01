from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier


class NaiveBayes:
    def start(self, train_df, test_df):
        # vectorizer = CountVectorizer(ngram_range=(1, 3))
        vectorizer = CountVectorizer(ngram_range=(1, 6))
        train_table = vectorizer.fit_transform(train_df['sentence'])
        test_table = vectorizer.transform(test_df['sentence'])

        model = MultinomialNB(alpha=0.1)
        model.fit(train_table, train_df['label'])
        predictions = model.predict(test_table)

        print(classification_report(test_df['label'], predictions.round()))
