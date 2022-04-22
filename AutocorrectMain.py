from time import time

from NgramAutocorrect import ngram_autocorrect, get_model
from NLPProcessing import open_single
from NaiveAutocorrect import naive_autocorrect


def correct_sentence(input, terms, total_words, n, model=None):
    corrected = []
    input = input.lower()
    sentence = input.split(" ")
    for i in range(len(sentence)):
        term = sentence[i]
        if term in terms.keys():
            corrected.append((1, term))
        else:
            if n == 1:
                corrected.append(naive_autocorrect(i, sentence, terms, total_words))
            else:
                corrected.append(ngram_autocorrect(i, sentence, model, terms, n))
        sentence[i] = corrected[len(corrected) - 1][1]
    return corrected


def user_input(terms, total_words, n, model=None):
    userInput = input("Enter a sentence with spelling errors to be corrected: ")
    while not userInput == "":
        cur_time = time()
        max_sum = 0
        output = correct_sentence(userInput, terms, total_words, n, model)
        # print(time() - cur_time)
        sentence = " ".join([w for (p, w) in output])
        print("Autocorrect thinks you meant: ", sentence)
        print("With probabilities: ", output)
        userInput = input("Enter a sentence with spelling errors to be corrected: ")

terms, total_words = open_single()
laplace_model = get_model('laplace_ngram_model.pkl')
wbi_model = get_model('wbi_ngram_model.pkl')
user_input(terms, total_words, 3, wbi_model)
#print(correct_sentence("Ths prigram auomaticaly fixs speling rrrora for the user somwht acuratly", terms, total_words))
