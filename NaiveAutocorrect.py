import concurrent.futures
import multiprocessing
import string

import nltk
from multiprocessing import Pool
import json
import threading
from time import time

import numpy as np
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('reuters')
nltk.download('punkt')
from nltk.corpus import reuters

alph = "abcdefghijklmnopqrstuvwxyz"

misspelled_word = "ztay"

weights = {"del_cost": 2, "ins_cost": 1, "rep_cost": 2}

dictionary = ["apple", "stay", "play"]

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

def process_ngrams():

    sentences = reuters.sents()
    words = reuters.words()
    trigram = {}
    fourgram = {}
    for sentence in sentences:
        tri = []
        four = []
        for word in sentence:
            if word not in string.punctuation:
                word = word.lower()
                if len(tri) < 3:
                    tri.append(str(word))
                elif len(tri) >= 3:
                    if tuple(tri) in trigram.keys():
                        trigram[tuple(tri)] += 1
                    else:
                        trigram[tuple(tri)] = 1
                if len(four) < 4:
                    four.append(str(word))
                elif len(four) >= 4:
                    if tuple(four) in fourgram.keys():
                        fourgram[tuple(four)] += 1
                    else:
                        fourgram[tuple(four)] = 1
    tri_strings = {}
    for tup in trigram.keys():
        tri_strings[" ".join(tup)] = trigram[tup]
    four_strings = {}
    for tup in fourgram.keys():
        four_strings[" ".join(tup)] = fourgram[tup]
    with open("trigrams.json", "w") as outfile:
        json.dump(tri_strings, outfile)
    with open("fourgrams.json", "w") as outfile:
        json.dump(four_strings, outfile)

def open_single():
    with open('processed_words.json') as json_file:
        terms = json.load(json_file)
    total_words = 0
    for term in terms.keys():
        total_words += terms[term]
    return terms, total_words


def open_ngram():
    with open('trigrams.json') as json_file:
        trigrams_list = json.load(json_file)
    trigrams = {}
    total_words = 0
    for trigram in trigrams_list:
        print(trigram)
        total_words += 3


def word_stats(term, terms, total_words):
    return terms[term] / total_words

def minEditDistance(source, target):
    dists = []
    for i in range(len(source) + 1):
        temp = [0] * (1 + len(target))
        dists.append(temp)
    for i in range(1, len(source) + 1):
        dists[i][0] = dists[i - 1][0] + weights["del_cost"]
    for i in range(1, len(target) + 1):
        dists[0][i] = dists[0][i - 1] + weights["ins_cost"]

    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            if source[i-1] == target[j-1]:
                dists[i][j] = dists[i - 1][j - 1]
            else:
                dists[i][j] = min(dists[i - 1][j] + weights["del_cost"], dists[i][j - 1] + weights["ins_cost"],
                                  dists[i - 1][j - 1] + weights["rep_cost"])
    return dists[len(source)][len(target)]

oneEditDistances = {}
def oneEditDistance(term):
    if term in oneEditDistances.keys():
        return oneEditDistances[term]
    splits = [(term[:i], term[i:]) for i in range(len(term) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    replaces = [L + c + R[1:] for L, R in splits if R for c in alph]
    inserts = [L + c + R for L, R in splits for c in alph]
    dict = []
    dict.extend(deletes)
    dict.extend(replaces)
    dict.extend(inserts)
    oneEditDistances[term] = set(dict)
    return oneEditDistances[term]

def nEditDistance(n, term, terms):
    dict = []
    prev_edits = oneEditDistance(term)
    for i in range(n - 1):
        prev_edits = [w2 for w1 in prev_edits for w2 in oneEditDistance(w1)]
        dict.extend(prev_edits)
        print(prev_edits)
    return set(w for w in dict if w in terms.keys())


def findSimilarWords(n, word, low, ret_words):
    for t in low:
        if abs(len(t) - len(word)) >= n:
            continue
        dist = minEditDistance(word, t)
        #print(t)
        if dist < n:
            ret_words.append((dist, t))
            #print(ret_words[:])
    return ret_words


def naive_autocorrect(term, terms, total_words):
    # potential = nEditDistance(2, term)
    n = 8
    potential = []#multiprocessing.Array('i', range(0))
    num_threads = 12
    words = np.array_split(list(terms.keys()), num_threads)
    threads = []
    pool = Pool(processes=num_threads)
    for i in range(num_threads):
        #print(words[i])
        x = pool.apply_async(findSimilarWords, (n, term, words[i], []))
        #x = threading.Thread(target=findSimilarWords, args=(n, term, words[i], potential))
        # print(potential[:])
        threads.append(x)
    # pool.close()
    # pool.join()
    #     x.start()
    for thread in threads:
        # thread.join()
        potential.extend(thread.get(timeout=1))
    # print("in autocorrect", potential)
    maxWordProb = 0
    bestWord = "???"
    maxWordDist = 5

    for (dist, t) in potential:
        prob = word_stats(t, terms, total_words)
        if prob > maxWordProb and maxWordDist > dist:
            maxWordProb = prob
            maxWordDist = dist
            bestWord = t
            #print("first: ", bestWord)
        elif maxWordDist > dist and prob < 50 / total_words:
            maxWordProb = prob
            maxWordDist = dist
            bestWord = t
            #print("2nd: ", bestWord)

        elif prob / pow(dist, dist) > maxWordProb / pow(maxWordDist, maxWordDist):
            maxWordProb = prob
            maxWordDist = dist
            bestWord = t
            #print("3rd: ", bestWord)

        #print(maxWordProb, maxWordDist, bestWord)
    return (maxWordProb, bestWord)

def correct_sentance(input, autocorrectfn, terms, total_words):
    corrected = []
    input = input.lower()
    for term in input.split(" "):
        if term in terms.keys():
            corrected.append((1, term))
        else:
            corrected.append(autocorrectfn(term, terms, total_words))
    return corrected


def user_input(terms, total_words):
    userInput = input("Enter a sentence with spelling errors to be corrected: ")
    while not userInput == "":
        cur_time = time()
        output = correct_sentance(userInput, naive_autocorrect, terms, total_words)
        # print(time() - cur_time)
        sentence = " ".join([w for (p, w) in output])
        print("Autocorrect thinks you meant: ", sentence)
        print("With probabilities: ", output)
        userInput = input("Enter a sentence with spelling errors to be corrected: ")


# print(minEditDistance(misspelled_word, dictionary[0]))
#process_original_json('ap201001.json')
#print(correct_sentance("This prigram automaticaly fixes spelling rrrora for th user somwht acuraty"))
terms, total_words = open_single()
user_input(terms, total_words)
#process_ngrams()
#open_ngram()

