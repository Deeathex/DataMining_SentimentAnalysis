from nltk.corpus import stopwords

from data_processing.text_processing.data_preprocessing import data_cleaning, clean_text_round1, clean_text_round2, \
    clean_text_remove_numbers, tokenization, lemmatization
from data_processing.text_processing.feature_extraction import feature_extraction
from data_processing.text_processing.utils import vizualize_topics

import pandas as pd
import gensim
from gensim import corpora
import nltk
nltk.download('stopwords')

def generate_corpus(data):
    corpus = []
    for index, row in data.iterrows():
        text = row['Text'].split(" ")
        corpus.append(text)
    return corpus


def topic_modeling(data_clean):
    corpus = generate_corpus(data_clean)
    dictionary = corpora.Dictionary(corpus)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=4, id2word=dictionary, passes=40)
    print(ldamodel.print_topics())
    sw = stopwords.words('english')
    vizualize_topics(ldamodel, sw)


def data_normalization_and_saving(data_features):
    data_numeric = data_features.copy()
    del data_numeric['Text']
    del data_numeric['Id']
    del data_numeric['Label']
    del data_numeric['Raw text']

    normalized_df_min_max = (data_numeric - data_numeric.min()) / (data_numeric.max() - data_numeric.min())

    normalized_df_min_max['Text'] = data_features['Text']
    normalized_df_min_max['Raw text'] = data_features['Raw text']
    normalized_df_min_max['Label'] = data_features['Label']
    normalized_df_min_max['Id'] = data_features['Id']

    normalized_df_min_max.to_pickle("data/features_norm_minmax.pkl")
    normalized_df_min_max.to_csv("data/features_norm_minmax.csv")


if __name__ == "__main__":
    dataframe = data_cleaning(1000)  # dataframe initial DATA
    round1 = lambda x: clean_text_round1(x)
    round2 = lambda x: clean_text_round2(x)
    round3 = lambda x: clean_text_remove_numbers(x)

    data_clean = pd.DataFrame(dataframe.Text.apply(round1))
    data_clean = pd.DataFrame(data_clean.Text.apply(round2))
    data_clean = pd.DataFrame(data_clean.Text.apply(round3))

    token = lambda x: tokenization(x)
    lemmas = lambda x: lemmatization(x)

    data_clean = pd.DataFrame(data_clean.Text.apply(token))
    corpus_without_lemmas = data_clean.copy(deep=True)

    data_clean = pd.DataFrame(data_clean.Text.apply(lemmas))
    corpus_with_lemmas = data_clean.copy(deep=True)

    data_clean['Id'] = dataframe['Id']
    data_clean['Label'] = dataframe['Label']

    data_features = feature_extraction(dataframe, data_clean, corpus_without_lemmas)

    data_normalization_and_saving(data_features)

    # topic_modeling(data_clean)
