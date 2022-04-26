from nltk.corpus import reuters
import json
import dill as pickle
from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.lm import MLE, Laplace, WittenBellInterpolated
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('reuters')
# nltk.download('punkt')
from nltk.corpus import reuters


def process_original_json(filepath):
    proc_terms = {}
    total_words = 0
    rawfile = open(filepath)
    terms = [json.loads(line) for line in rawfile]
    for term in terms:
        total_words += term["count"]
        if term["term"] in proc_terms.keys():
            proc_terms[term["term"]] += term["count"]
        else:
            proc_terms[term["term"]] = term["count"]
    words = sorted(proc_terms.keys(), key=len)
    final_terms = {}
    for w in words:
        final_terms[w] = proc_terms[w]
    with open("processed_words.json", "w") as outfile:
        json.dump(final_terms, outfile)

#process_original_json('ap201001.json')


def open_single():
    with open('venv/unigrams.json') as json_file:
        terms = json.load(json_file)
    total_words = 0
    for term in terms.keys():
        total_words += terms[term]
    return terms, total_words


def word_stats(term, terms, total_words):
    return terms[term] / total_words


def train_hitler_ai():
    text = ""
    with open('venv/meinkampf.txt') as txt:
        text = txt.read()
    tokenized_text = [list(map(str.lower, word_tokenize(sent)))
                      for sent in sent_tokenize(text)]
    n = 7
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
    model = WittenBellInterpolated(n)
    model.fit(train_data, padded_sents)
    with open('hitler_ngram_model.pkl', 'wb') as fout:
        pickle.dump(model, fout)


def train_model():
    text = ""
    for sent in reuters.sents():
        for word in sent:
            if word.isalpha() and len(word) >= 1:
                text += " " + (word.lower())
    tokenized_text = [list(map(str.lower, word_tokenize(sent)))
                      for sent in sent_tokenize(text)]
    train_data, padded_sents = padded_everygram_pipeline(2, tokenized_text)
    model = MLE(2)
    model.fit(train_data, padded_sents)
    with open('bigram_model.pkl', 'wb') as fout:
        pickle.dump(model, fout)


def train_model_general(filepath, endfilepath):
    text = ""
    with open(filepath) as txt:
        text = txt.read()
    tokenized_text = [list(map(str.lower, word_tokenize(sent)))
                      for sent in sent_tokenize(text)]
    with open('venv/unigrams.json') as json_file:
        terms = json.load(json_file)
    for sent in tokenized_text:
        for term in sent:
            if term in terms.keys():
                terms[term] += 1
            else:
                terms[term] = 1
    with open("venv/unigrams.json", "w") as outfile:
        json.dump(terms, outfile)
    n = 3
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
    model = Laplace(n)
    model.fit(train_data, padded_sents)
    with open(endfilepath, 'wb') as fout:
        pickle.dump(model, fout)

#train_model()
#train_model_general('venv/en_US.twitter.txt', 'twitter_trigram_model.pkl')