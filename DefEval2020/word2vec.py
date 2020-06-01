import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
from gensim.models import word2vec
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model as lm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class Word2vec:

    def sentence_to_wordlist(self,sentence):
        # to Remove HTML
        clean_sentence = BeautifulSoup(sentence).get_text()
        # to Remove non-letters
        clean_sentence = re.sub("[^a-zA-Z]", " ", clean_sentence)
        # Convert words to lower case and split them
        words = clean_sentence.lower().split()

        return (words)

    def data_to_sentences(self,data):
        sentences_array = []
        for raw_sentence in data['sentence']:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences_array.append(self.sentence_to_wordlist(raw_sentence))

        return sentences_array

    def words_avg(self,words, model, num_features):
        featureVec = np.zeros((num_features,), dtype="float32")

        no_words = 0

        index2word_set = set(model.wv.index2word)

        for word in words:
            if word in index2word_set:
                no_words = no_words + 1
                featureVec = np.add(featureVec, model[word])

        featureVec = np.divide(featureVec, no_words)
        return featureVec

    def getAvgFeatureVecs(self,sentences, model, num_features):

        counter = 0
        FeatureVecs = np.zeros((len(sentences), num_features), dtype="float32")

        for sentence in sentences:
            FeatureVecs[counter] = self.words_avg(sentence, model, num_features)
            counter = counter + 1
        return FeatureVecs

    def start(self):

        # Read data from files
        train = pd.read_table('all_in_one_train.deft', sep='\t', header=None,
                              names=['sentence', 'label'])
        test = pd.read_csv('all_in_one_dev.deft', sep='\t', header=None,
                           names=['sentence', 'label'])

        # to check the number of sentences
        # print ( train["sentence"].size, test["sentence"].size )

        train_sentences = self.data_to_sentences(train)

        num_features = 300  # Word vector dimensionality
        min_word_count = 10  # Minimum word count
        num_workers = 4  # Number of threads to run in parallel
        context = 10  # Context window size

        model = word2vec.Word2Vec(train_sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                                  window=context)

        model.init_sims(replace=True)
        # model.most_similar("biology")  for testing the model

        clean_train = []
        for sentence in train["sentence"]:
            clean_train.append(self.sentence_to_wordlist(sentence))

        trainDataVecs = self.getAvgFeatureVecs(clean_train, model, num_features)

        clean_test = []
        for sentence in test["sentence"]:
            clean_test.append(self.sentence_to_wordlist(sentence))

        testDataVecs = self.getAvgFeatureVecs(clean_test, model, num_features)
        print("--> Using Word2vec")
        # random forest (just for testing)
        print("random forest (just for testing)..")
        forest = RandomForestClassifier(n_estimators=100)
        forest = forest.fit(np.nan_to_num(trainDataVecs), train["label"])
        result = forest.predict(np.nan_to_num(testDataVecs))
        print(classification_report(test['label'], result.round()))

        # logistic regression
        print("logistic regression..")
        logistic = lm.LogisticRegression()
        logistic.fit(np.nan_to_num(trainDataVecs), train['label'])
        predictions = logistic.predict(np.nan_to_num(testDataVecs))
        print(classification_report(test['label'], predictions.round()))

        # decision tree
        print("decision tree..")
        tree = DecisionTreeClassifier()
        tree.fit(np.nan_to_num(trainDataVecs), train['label'])
        predictions = tree.predict(np.nan_to_num(testDataVecs))
        print(classification_report(test['label'], predictions.round()))

        # K-nearest neighbours
        print("K-nearest neighbours..")
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(np.nan_to_num(trainDataVecs), train['label'])
        predictions = knn.predict(np.nan_to_num(testDataVecs))
        print(classification_report(test['label'], predictions.round()))