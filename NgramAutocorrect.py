import dill as pickle
import difflib
import ssl

import nltk
from nltk.metrics.distance import jaro_winkler_similarity
from nltk.tokenize.treebank import TreebankWordDetokenizer

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('reuters')
# nltk.download('punkt')
from nltk.corpus import reuters



def get_model(filepath):
    with open(filepath, 'rb') as fin:
        model = pickle.load(fin)
    return model


def ngram_autocorrect(index, context, model, terms, n):
    term = context[index]
    similar = [t for t in terms.keys() if jaro_winkler_similarity(t, term, p=0.1) > .88]

    print(similar)
    maxWordProb = float('-inf')
    bestWord = term
    for word in set(similar):
        temp = n
        sentence = context[max((index - n), 0):index]
        #print(sentence, word)
        while len(sentence) > 0 and model.counts[sentence][word] == 0:
            temp -= 1
            sentence = context[index - temp:index]
            print(sentence)
        prob = model.logscore(word, sentence)

        if prob > maxWordProb:
            maxWordProb = prob
            bestWord = word
    return maxWordProb, bestWord

def bigram_autocorrect(index, context, model, terms):
    term = context[index]
    cf_biag = nltk.ConditionalFreqDist(model)
    cf_biag = nltk.ConditionalProbDist(cf_biag, nltk.MLEProbDist)
    similar = [t for t in terms.keys() if jaro_winkler_similarity(t, term, p=0.1) > .88]
    if len(similar) < 2:
        similar = difflib.get_close_matches(term, terms.keys(), 5)
    i = 0
    for word in similar:
        print()



detokenize = TreebankWordDetokenizer().detokenize


def generate_sent(model, num_words, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)


# train_model()
#real_model = get_model('ngram_model.pkl')
#train_hitler_ai()
#hitler_model = get_model('hitler_ngram_model.pkl')

#unigram, total_words = open_single()
#print(ngram_autocorrect(2, ['last', 'year', 'th'], real_model, unigram))
# print(model.logscore('linguistics', 'journal of'.split()))
# print(modl.score('never', 'language is'.split()))
# print(modl.score('fear', "asian exporters".split()))

# print(model.counts)
# print(real_model.generate(100))
#print(generate_sent(hitler_model, 200, random_seed=2))

#print(real_model.score('the', 'last year'.split()))


# print(get_ngrams(reuters.sents(), 3).counts['asian'])
