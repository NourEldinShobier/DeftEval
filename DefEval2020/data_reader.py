import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class DataReader:

    def read_file(self, file_name: str):
        stop_words = stopwords.words('english')
        stemmer = PorterStemmer()

        data_frame = pd.read_table(file_name, sep='\t', header=None, names=['sentence', 'label'])

        data_frame['sentence'] = data_frame.sentence.map(lambda x: x.lower())
        data_frame['sentence'] = data_frame.sentence.str.replace('[^\w\s]', '')

        data_frame['sentence'] = data_frame['sentence'].apply(nltk.word_tokenize)

        data_frame['sentence'] = data_frame['sentence'].apply(lambda x: [word for word in x if word not in stop_words])
        data_frame['sentence'] = data_frame['sentence'].apply(lambda x: [stemmer.stem(y) for y in x])

        data_frame['sentence'] = data_frame['sentence'].apply(lambda x: ' '.join(x))

        return data_frame
