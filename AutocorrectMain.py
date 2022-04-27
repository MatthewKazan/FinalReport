from time import time

from NgramAutocorrect import ngram_autocorrect, get_model, bigram_autocorrect, generate_sent
from NLPProcessing import open_single, train_hitler_ai
from NaiveAutocorrect import naive_autocorrect


def correct_sentence(input, terms, total_words, n, model=None):
    corrected = []
    input = input.lower()
    sentence = input.split(" ")
    for i in range(len(sentence)):
        term = sentence[i]
        if term in terms.keys():
            corrected.append((1, term))
        elif term.isalpha():
            if n == 1:
                corrected.append(naive_autocorrect(i, sentence, terms, total_words))
            else:
                corrected.append(ngram_autocorrect(i, sentence, model, terms, n))
        else:
            corrected.append((0, term))
        sentence[i] = corrected[len(corrected) - 1][1]
    return corrected


def user_input(terms, total_words, n, model=None):
    userInput = input("Enter a sentence with spelling errors to be corrected: ")
    while not userInput == "":
        cur_time = time()
        output = correct_sentence(userInput, terms, total_words, n, model)
        # print(time() - cur_time)
        sentence = " ".join([w for (p, w) in output])
        print("Autocorrect thinks you meant: ", sentence)
        print("With probabilities: ", output)
        userInput = input("Enter a sentence with spelling errors to be corrected: ")


twitter_terms, twitter_total_words = open_single('twitter_unigrams.json')
reuters_terms, reuters_total_words = open_single('reuters_unigrams.json')

# hitler_model = get_model('OldModels/hitler_ngram_model.pkl')
twitter_bigram_model = get_model('twitter_bigram_laplace.pkl')
twitter_trigram_model = get_model('twitter_trigram_laplace.pkl')
reuters_trigram_model = get_model('reuters_trigram_laplace.pkl')

# print(generate_sent(hitler_model, 20000))
# print(generate_sent(twitter_model, 200))
#user_input(twitter_terms, twitter_total_words, 1, twitter_trigram_model)
print(correct_sentence("Ths prigram auomaticaly fixs speling istks for the user somwht acuratly", twitter_terms,
                       twitter_total_words, 3, twitter_trigram_model))
print(correct_sentence("Both Microsoft and Yahoo ! began yscissig succ an arlangement .", twitter_terms,
                       twitter_total_words, 3, twitter_trigram_model))
