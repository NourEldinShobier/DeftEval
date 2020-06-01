from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model as lm
model = lm.LogisticRegression()




class LogisticRegression:
    def start(self, train_df, test_df):
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        train_table = vectorizer.fit_transform(train_df['sentence'])
        test_table = vectorizer.transform(test_df['sentence'])

        model.fit(train_table, train_df['label'])
        predictions = model.predict(test_table)
        print(classification_report(test_df['label'], predictions.round()))
