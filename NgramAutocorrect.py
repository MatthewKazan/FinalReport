import dill as pickle
from nltk.metrics.distance import jaro_winkler_similarity, jaccard_distance, edit_distance, jaro_similarity
from nltk.tokenize.treebank import TreebankWordDetokenizer


def get_model(filepath):
    with open(filepath, 'rb') as fin:
        model = pickle.load(fin)
    return model


def ngram_autocorrect(index, context, model, terms, n):
    term = context[index]
    similar = [(jaro_winkler_similarity(t, term, p=0.1), t) for t in terms.keys() if jaro_winkler_similarity(t, term, p=0.1) > .8]

    maxProb = float('-inf')
    minProb = float('inf')
    probs = {}
    bestWord = term
    for (dist, t) in similar:
        prob = 0
        temp = (context.copy())
        for x in range(1, n):
            if index - x >= 0:
                prob += (model.logscore(t, temp[index - x:index])) * (n - x)
        prob += model.logscore(t) / 5
        prob /= n
        probs[t] = prob
        minProb = min(minProb, prob)
        maxProb = max(maxProb, prob)
    maxWordProb = 0
    for (dist, t) in similar:
        prob = (probs[t] - minProb) / ((maxProb + 1) - minProb)
        prob /= 7
        # print(t, prob)
        prob += dist
        # print(t, prob, dist)

        if prob > maxWordProb:
            maxWordProb = prob
            bestWord = t

    return maxWordProb, bestWord


def bigram_autocorrect(index, context, model, terms):
    term = context[index]
    similar = [t for t in terms.keys() if jaro_winkler_similarity(t, term, p=0.1) > .8]
    maxWordProb = float('-inf')
    bestWord = term

    for word in set(similar):
        pre = ""
        post = ""
        if index > 0:
            pre = context[index - 1]
        if index < len(context) - 1:
            post = context[index + 1]
        prob = (model.score(word, pre.split()))
        prob *= (model.score(post, [word]))
        prob *= model.score(word)
        if prob > maxWordProb:
            maxWordProb = prob
            bestWord = word
    return maxWordProb, bestWord


detokenize = TreebankWordDetokenizer().detokenize


def generate_sent(model, num_words):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    """
    content = []
    for token in model.generate(num_words):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)


#real_model = get_model('bigram_model.pkl')
#


#print(ngram_autocorrect(2, ['last', 'year', 'th'], real_model, unigram))
# print(model.logscore('linguistics', 'journal of'.split()))
# print(modl.score('never', 'language is'.split()))
# print(modl.score('fear', "asian exporters".split()))

# print(real_model.generate(100))
#print(generate_sent(hitler_model, 200, random_seed=2))

#print(real_model.score('the', 'last year'.split()))
#unigram, total_words = open_single()
#print(bigram_autocorrect(2, 'this last yer the speling rrrora for the user somwht acuratly'.split(), real_model, unigram))
#print(ngram_autocorrect(2, 'this last yer the speling rrrora for the user somwht acuratly'.split(), real_model, unigram, 2))
# print(get_ngrams(reuters.sents(), 3).counts['asian'])
