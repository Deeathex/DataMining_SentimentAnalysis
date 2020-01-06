import pandas as pd
import nltk
import re
from nltk.stem.porter import PorterStemmer


class AbstractDataset:
    def __init__(self, path_to_csv, separator=','):
        self.__path_to_csv = path_to_csv
        self.__separator = separator
        self._raw_data = None
        self.__read_csv()

    def __read_csv(self):
        # display all columns from the dataset if ever printed
        pd.options.display.width = 0

        # read the data from the csv
        self._raw_data = pd.read_csv(self.__path_to_csv, sep=self.__separator, nrows=15000)

    def get_preprocessed_rows_text(self, text_column=0):
        # take the text of each row in 'rows_text'
        rows_text = self._raw_data.iloc[:, text_column].values

        # do the general preprocessing
        processed_rows = []
        ps = PorterStemmer()
        for i in range(0, len(rows_text)):
            # remove all the special characters (@,!,#,$ etc)
            processed_row = re.sub(r'[^0-9a-zA-Z\'\s]+', ' ', str(rows_text[i]))

            # remove 1 letter words
            processed_row = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_row)

            # remove 1 letter words from the beginning
            processed_row = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_row)

            # replace multiple spaces with a single one
            processed_row = re.sub(r'\s+', ' ', processed_row)

            # apply some PorterStemmer on the row
            # processed_row = [ps.stem(word) for word in processed_row.split(' ')]
            # processed_row = ' '.join(processed_row)

            # convert tweet to lowercase and store it
            processed_rows.append(processed_row.lower())
        return processed_rows

    def get_preprocessed_rows_sentiment(self, sentiment_column=1):
        # take the sentiment value of each row in 'rows_sentiment'
        rows_sentiment = self._raw_data.iloc[:, sentiment_column].values
        return rows_sentiment

    def get_stop_words(self):
        # download the stopwords if required
        nltk.download('stopwords')

        # return them
        return nltk.corpus.stopwords.words('english')

    def display_data_metrics(self):
        pass
