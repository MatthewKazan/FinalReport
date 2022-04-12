import json

alph = "abcdefghijklmnopqrstuvwxyz"

misspelled_word = "ztay"

weights = {"del_cost": 1, "ins_cost": 1, "rep_cost": 2}

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
    with open("processed_words.json", "w") as outfile:
        json.dump(proc_terms, outfile)

process_original_json('ap201001.json')

with open('processed_words.json') as json_file:
    terms = json.load(json_file)

total_words = 0
for term in terms.keys():
    total_words += terms[term]


def word_stats(term):
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

def oneEditDistance(term):
    splits = [(term[:i], term[i:]) for i in range(len(term) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    replaces = [L + c + R[1:] for L, R in splits if R for c in alph]
    inserts = [L + c + R for L, R in splits for c in alph]
    dict = []
    dict.extend(deletes)
    dict.extend(replaces)
    dict.extend(inserts)
    return dict

def nEditDistance(n, term):
    dict = []
    prev_edits = oneEditDistance(term)
    for i in range(n - 1):
        dict.extend(set(w2 for w1 in prev_edits for w2 in oneEditDistance(w1)))
    return set(w for w in dict if w in terms.keys())

def naive_autocorrect(term):
    potential = nEditDistance(3, term)
    maxWordProb = 0
    bestWord = ""
    for term in potential:
        prob = word_stats(term)
        if prob > maxWordProb:
            maxWordProb = prob
            bestWord = term
    return (maxWordProb, bestWord)

def correct_sentance(input):
    corrected = []
    input = input.lower()
    for term in input.split(" "):
        if term in terms.keys():
            corrected.append((1, term))
        else:
            corrected.append(naive_autocorrect(term))
    return corrected
# print(minEditDistance(misspelled_word, dictionary[0]))
# process_original_json('ap201001.json')
# print(word_stats("a"))
# print(nEditDistance(4, "somthing"))


#print(correct_sentance("This prigram automaticaly fixes spelling rrrora for th user somwht acuraty"))
userInput = input("Enter a sentence with spelling errors to be corrected: ")
sentence = " ".join([w for (p,w) in correct_sentance(userInput)])
print("The program thinks you meant: ", sentence)
print("With probabilities: ", correct_sentance(userInput))