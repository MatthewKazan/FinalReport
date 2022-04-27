from multiprocessing import Pool
import numpy as np

from NLPProcessing import word_stats

alph = "abcdefghijklmnopqrstuvwxyz"

weights = {"del_cost": 2, "ins_cost": 1, "rep_cost": 2}


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
    return set(w for w in dict if w in terms.keys())


def findSimilarWords(n, word, low, ret_words):
    for t in low:
        if abs(len(t) - len(word)) >= n:
            continue
        dist = minEditDistance(word, t)
        if dist < n:
            ret_words.append((dist, t))
    return ret_words


def naive_autocorrect(index, sentence, terms, total_words):
    term = sentence[index]
    n = len(term) - 1
    potential = []
    num_threads = 12
    words = np.array_split(list(terms.keys()), num_threads)
    threads = []
    # pool = Pool(processes=num_threads)
    # for i in range(num_threads):
    #     # print(words[i])
    #     x = pool.apply_async(findSimilarWords, (n, term, words[i], []))
    #     threads.append(x)
    # for thread in threads:
    #     potential.extend(thread.get(timeout=1))
    potential = findSimilarWords(n, term, terms.keys(), [])
    maxWordProb = 0
    bestWord = term
    maxWordDist = 5
    for (dist, t) in potential:
        prob = word_stats(t, terms, total_words)
        if prob > maxWordProb and maxWordDist > dist:
            maxWordProb = prob
            maxWordDist = dist
            bestWord = t
            # print("first: ", bestWord)
        elif maxWordDist > dist and abs(maxWordProb - prob) < 50 / total_words:
            maxWordProb = prob
            maxWordDist = dist
            bestWord = t
            # print("2nd: ", bestWord)

        elif prob / pow(dist, dist) > maxWordProb / pow(maxWordDist, maxWordDist):
            maxWordProb = prob
            maxWordDist = dist
            bestWord = t
            # print("3rd: ", bestWord)

        # print(maxWordProb, maxWordDist, bestWord)
    return (maxWordProb, bestWord)





# print(minEditDistance(misspelled_word, dictionary[0]))

#unigrams = open_ngram()
#print(single_word_prob("pineapples", "the u s", trigrams, fourgrams, total_words))
#print(correct_sentance("Ths prigram auomaticaly fixs speling rrrora for the user somwht acuratly", naive_autocorrect, unigrams, total_words))
#print(correct_sentance("Ths prigram auomaticaly fixs speling rrrora for the user somwht acuratly", ngram_autocorrect, unigrams, total_words, trigrams, fourgrams))


