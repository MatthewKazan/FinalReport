from nltk.corpus import reuters
import json
import dill as pickle
from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.lm import MLE, Laplace, WittenBellInterpolated


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
    n = 3
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
    model = MLE(3)  # Lets train a 3-grams maximum likelihood estimation model.
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
    train_data, padded_sents = padded_everygram_pipeline(3, tokenized_text)
    model = WittenBellInterpolated(3)
    model.fit(train_data, padded_sents)
    with open('wbi_ngram_model.pkl', 'wb') as fout:
        pickle.dump(model, fout)


#train_model()
