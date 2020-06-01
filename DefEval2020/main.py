from DefEval2020.naive_bayes import NaiveBayes
from DefEval2020.decision_tree import DecisionTree
from DefEval2020.knn import KNN
from DefEval2020.data_reader import DataReader
from DefEval2020.logisticRegression import LogisticRegression
from DefEval2020.word2vec import Word2vec
from DefEval2020.data_analyser import DataAnalyser


def main():
    data_reader = DataReader()
    train_df = data_reader.read_file('all_in_one_train.deft')
    test_df = data_reader.read_file('all_in_one_dev.deft')

    print('Naive Bayes..')
    naive_bayes = NaiveBayes()
    naive_bayes.start(train_df, test_df)

    print('decision tree..')
    decision_tree = DecisionTree()
    decision_tree.start(train_df, test_df)

    print('K-Nearest Nieghbors..')
    knn = KNN()
    knn.start(train_df, test_df)

    print('Logistic Regression..')
    knn = LogisticRegression()
    knn.start(train_df, test_df)

    word2vec = Word2vec()
    word2vec.start()

main()
