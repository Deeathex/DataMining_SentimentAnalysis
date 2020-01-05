import pandas as pd
from data_processing.text_processing.utils import generateId

import re
import string


def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text


# Apply a second round of cleaning
def clean_text_remove_numbers(text):
    '''Get rid of numbers encountered in text'''
    text = re.sub(r'[0-9]+', '', text)
    ''' Get rid of hashtags or persons tagged'''
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", text).split())
    ''' Get rid of websides if they exists'''
    if 'www.' in text or 'http:' in text or 'https:' in text or '.com' in text:
        text = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "website", text)
    return text


def data_cleaning(no_samples_per_class):
    Data = {
        'Id': [],
        'Text': [],
        'Label': []
    }
    pozitives = 0
    negatives = 0
    # 1. Load data
    with open("D:\\MASTER\\SEMESTRUL1\\DataMining\\DataSets\\amazonreviews\\train.ft.txt", encoding='utf-8') as infile:
        for line in infile:
            label = line.split(" ")[0]
            review = " ".join(line.split(" ")[1:])
            # 2. Insert data into pandas dataframe
            if (review.strip() == ""):
                continue
            if label == '__label__2':
                # reduce data from technical reason
                if (pozitives < no_samples_per_class):
                    Data['Label'].append('pozitive')
                    Data['Id'].append(generateId())
                    Data['Text'].append(review)
                    pozitives += 1
            else:
                # reduce data from technical reason
                if (negatives < no_samples_per_class):
                    Data['Label'].append('negative')
                    Data['Id'].append(generateId())
                    Data['Text'].append(review)
                    negatives += 1

    dataframe = pd.DataFrame(Data, columns=['Id', 'Text', 'Label'])
    return dataframe


from nltk.tokenize import TreebankWordTokenizer;
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Init Lemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# Apply tokenization and lemmatization and removing stop words
def tokenization(text):
    '''Tokenization of the text'''
    tokens = TreebankWordTokenizer().tokenize(text)
    filtered_text = [w for w in tokens if not w in stop_words]
    text = " ".join(filtered_text)
    #     print(text)
    return text


def lemmatization(text):
    ''' Lemmatization of the text'''
    tokens = text.split(" ")
    lemmas = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    filtered_text = [w for w in lemmas if not w in stop_words]
    text = " ".join(filtered_text)
    #     print(text)
    return text
