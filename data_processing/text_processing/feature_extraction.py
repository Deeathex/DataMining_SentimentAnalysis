import string
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('brown')


def get_no_punctuation_signs(text):
    k = 0
    for i in text:
        # checking whether the char is punctuation.
        if i in string.punctuation:
            k += 1
    return k


def get_no_words_capitalized(text):
    k = 0;
    words = text.split(" ")
    for word in words:
        chars = list(word)
        if (len(chars) > 0 and chars[0].isupper()):
            k += 1
    return k


def get_no_mark(text, mark):
    k = 0
    for i in text:
        if i in mark:
            k += 1
    return k


def get_no_words(text):
    words = text.split(" ")
    return len(words)


def get_no_pozitive_words(text):
    k = 0
    words = text.split(" ")
    for word in words:
        blob = TextBlob(word)
        if (blob.sentiment.polarity >= 0.1):
            k += 1
    return k


def get_no_negative_words(text):
    k = 0
    words = text.split(" ")
    for word in words:
        blob = TextBlob(word)
        if (blob.sentiment.polarity <= -0.1):
            k += 1
    return k


def get_pozitivity(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


def get_subjectivity(text):
    blob = TextBlob(text)
    return blob.sentiment.subjectivity


def identify_hashtags(text):
    result = []
    words = text.split(" ")
    for word in words:
        chars = list(word)
        if len(chars) > 0 and chars[0] == '#':
            result.append("".join(chars[1:]))
    return result


def get_no_hashtags(text):
    return len(identify_hashtags(text))


def get_no_poz_hashtags(text):
    hashtags = identify_hashtags(text)
    return get_no_pozitive_words(" ".join(hashtags))


def get_no_neg_hashtags(text):
    hashtags = identify_hashtags(text)
    return get_no_negative_words(" ".join(hashtags))


def get_no_poz_words_capitalized(text):
    k = 0;
    words = text.split(" ")
    for word in words:
        chars = list(word)
        if (len(chars) > 0 and chars[0].isupper()):
            blob = TextBlob(word)
            if (blob.sentiment.polarity >= 0.1):
                k += 1
    return k


def get_no_neg_words_capitalized(text):
    k = 0;
    words = text.split(" ")
    for word in words:
        chars = list(word)
        if (len(chars) > 0 and chars[0].isupper()):
            blob = TextBlob(word)
            if (blob.sentiment.polarity <= -0.1):
                k += 1
    return k


def get_no_urls(text):
    k = 0
    words = text.split(" ")
    for word in words:
        if 'http:' in word or 'https:' in word or 'www.' in word or '.com' in word:
            k += 1
    return k


def get_percentage_capitalized_words(text):
    return get_no_words_capitalized(text) / get_no_words(text)


def get_words_for_tag(text, tag):
    result = []
    blob = TextBlob(text)
    for entry in blob.tags:
        if (entry[1].startswith(tag)):
            result.append(entry[0])
    return result


def get_no_poz_nouns(text):
    nouns = get_words_for_tag(text, 'NN')
    return get_no_pozitive_words(" ".join(nouns))


def get_no_poz_adjectives(text):
    adjs = get_words_for_tag(text, 'JJ')
    return get_no_pozitive_words(" ".join(adjs))


def get_no_poz_adverbs(text):
    advs = get_words_for_tag(text, 'RB')
    return get_no_pozitive_words(" ".join(advs))


def get_no_poz_verbs(text):
    vbs = get_words_for_tag(text, 'VB')
    return get_no_pozitive_words(" ".join(vbs))


def get_no_neg_nouns(text):
    nouns = get_words_for_tag(text, 'NN')
    return get_no_negative_words(" ".join(nouns))


def get_no_neg_adjectives(text):
    adjs = get_words_for_tag(text, 'JJ')
    return get_no_negative_words(" ".join(adjs))


def get_no_neg_adverbs(text):
    advs = get_words_for_tag(text, 'RB')
    return get_no_negative_words(" ".join(advs))


def get_no_neg_verbs(text):
    vbs = get_words_for_tag(text, 'VB')
    return get_no_negative_words(" ".join(vbs))


def get_sum_polarity_words(text):
    words = text.split(" ")
    suma = 0
    for word in words:
        blob = TextBlob(word)
        suma += blob.sentiment.polarity
    return suma


def get_sum_polarity_nouns(text):
    word_tags = get_words_for_tag(text, 'NN')
    return get_sum_polarity_words(" ".join(word_tags))


def get_sum_polarity_verbs(text):
    word_tags = get_words_for_tag(text, 'VB')
    return get_sum_polarity_words(" ".join(word_tags))


def get_sum_polarity_adverbs(text):
    word_tags = get_words_for_tag(text, 'RB')
    return get_sum_polarity_words(" ".join(word_tags))


def get_sum_polarity_adjectives(text):
    word_tags = get_words_for_tag(text, 'JJ')
    return get_sum_polarity_words(" ".join(word_tags))

def feature_extraction(initial_data, data_clean, data_without_lemmas):
    data_features = data_clean.copy()

    arrayNoPunctuationSigns = []
    arrayNoWordsCapitalized = []
    arrayNoExclamationMark = []
    arrayNoQuestionMark = []
    arrayNoWords = []
    arrayPolarity = []
    arraySubjectivity = []
    arrayNoPozitiveWords = []
    arrayNoNegativeWords = []
    arrayNoHashtags = []
    arrayNoPozHashtags = []
    arrayNoNegHashtags = []
    arrayNoPozCapitalizedWords = []
    arrayNoNegCapitalizedWords = []
    arrayNoUrls = []
    arrayPercentageCapitalizedWords = []

    arrayNoPozNouns = []
    arrayNoNegNouns = []
    arrayNoPozVerbs = []
    arrayNoNegVerbs = []
    arrayNoNegAdverbs = []
    arrayNoPozAdverbs = []
    arrayNoPozAdjectives = []
    arrayNoNegAdjectives = []

    arraySumPolarityNouns = []
    arraySumPolarityVerbs = []
    arraySumPolarityAdverbs = []
    arraySumPolarityAdjectives = []

    for index, row in data_clean.iterrows():
        arrayNoPozitiveWords.append(get_no_pozitive_words(row['Text']))
        arrayNoNegativeWords.append(get_no_negative_words(row['Text']))


    for index, row in data_without_lemmas.iterrows():
        arrayNoPozNouns.append(get_no_poz_nouns(row['Text']))
        arrayNoNegNouns.append(get_no_neg_nouns(row['Text']))
        arrayNoPozVerbs.append(get_no_poz_verbs(row['Text']))
        arrayNoNegVerbs.append(get_no_neg_verbs(row['Text']))
        arrayNoPozAdverbs.append(get_no_poz_adverbs(row['Text']))
        arrayNoNegAdverbs.append(get_no_neg_adverbs(row['Text']))
        arrayNoPozAdjectives.append(get_no_poz_adjectives(row['Text']))
        arrayNoNegAdjectives.append(get_no_neg_adjectives(row['Text']))

        arraySumPolarityNouns.append(get_sum_polarity_nouns(row['Text']))
        arraySumPolarityVerbs.append(get_sum_polarity_verbs(row['Text']))
        arraySumPolarityAdjectives.append(get_sum_polarity_adjectives(row['Text']))
        arraySumPolarityAdverbs.append(get_sum_polarity_adverbs(row['Text']))

    for index, row in initial_data.iterrows():
        arrayNoPunctuationSigns.append(get_no_punctuation_signs(row['Text']))
        arrayNoWordsCapitalized.append(get_no_words_capitalized(row['Text']))
        arrayNoExclamationMark.append(get_no_mark(row['Text'], '!'))
        arrayNoQuestionMark.append(get_no_mark(row['Text'], '?'))
        arrayNoWords.append(get_no_words(row['Text']))
        arrayPolarity.append(get_pozitivity(row['Text']))
        arraySubjectivity.append(get_subjectivity(row['Text']))
        arrayNoHashtags.append(get_no_hashtags(row['Text']))
        arrayNoPozHashtags.append(get_no_poz_hashtags(row['Text']))
        arrayNoNegHashtags.append(get_no_neg_hashtags(row['Text']))
        arrayNoPozCapitalizedWords.append(get_no_poz_words_capitalized(row['Text']))
        arrayNoNegCapitalizedWords.append(get_no_neg_words_capitalized(row['Text']))
        arrayNoUrls.append(get_no_urls(row['Text']))
        arrayPercentageCapitalizedWords.append(get_percentage_capitalized_words(row['Text']))

    data_features['Number of punctuation signs'] = arrayNoPunctuationSigns
    data_features['Number of words capitalized'] = arrayNoWordsCapitalized
    data_features['Number of !'] = arrayNoExclamationMark
    data_features['Number of ?'] = arrayNoQuestionMark
    data_features['Number of words'] = arrayNoWords
    data_features['Overall polarity'] = arrayPolarity
    data_features['Overall subjectivity'] = arraySubjectivity
    data_features['Number of pozitive words'] = arrayNoPozitiveWords
    data_features['Number of negative words'] = arrayNoNegativeWords
    data_features['Number of hashtags'] = arrayNoHashtags
    data_features['Number of pozitive hashtags'] = arrayNoPozHashtags
    data_features['Number of negative hashtags'] = arrayNoNegHashtags
    data_features['Number of negative capitalized words'] = arrayNoNegCapitalizedWords
    data_features['Number of pozitive capitalized words'] = arrayNoPozCapitalizedWords
    data_features['Number of urls'] = arrayNoUrls
    data_features['Percentage of capitalized words'] = arrayPercentageCapitalizedWords
    data_features['Number of pozitive nouns'] = arrayNoPozNouns
    data_features['Number of negative nouns'] = arrayNoNegNouns
    data_features['Number of pozitive verbs'] = arrayNoPozVerbs
    data_features['Number of negative verbs'] = arrayNoNegVerbs
    data_features['Number of negative adverbs'] = arrayNoNegAdverbs
    data_features['Number of pozitive adverbs'] = arrayNoPozAdverbs
    data_features['Number of pozitive adjectives'] = arrayNoPozAdjectives
    data_features['Number of negative adjectives'] = arrayNoNegAdjectives
    data_features['Sum of all nouns polarity'] = arraySumPolarityNouns
    data_features['Sum of all verbs polarity'] = arraySumPolarityVerbs
    data_features['Sum of all adjectives polarity'] = arraySumPolarityAdjectives
    data_features['Sum of all adverbs polarity'] = arraySumPolarityAdverbs

    data_features['Raw text'] = initial_data['Text']

    return data_features