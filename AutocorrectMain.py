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
simpleInput = ["waffles and pancakrs", "hello worls", "spelling erris", "red rabbut", "basketball and volelyball"]
commonInput = ["nice to medt you", "where are you frpn", "what is your phone numbsa", "i am surryo", "excqse me"]
complexInput = ["tner a sentce with spelldng errurs to b correcsste", "so the horrbily outnumbers mongouse makes a desprta mve", "I was not expctng anyuns toda", "synnonyms and anytoonyms are oppisites"]
sentenceInput = ["I arrived at the bus station early but waited until noon for the bys"]
mumboJumboInput = ["asdfhasdf", "aosd ;mlkhloi aoijgres"]
inputs = [simpleInput, commonInput, complexInput, sentenceInput, mumboJumboInput]

longInput = ["she wrote him a lettr", "he liked to look at paintihs", "the fog was densf", "I looked in the mirrsr", "nobody loves a pig wearing lipstkc", "they ran around the cornsr",  "I want more detailed informatnsd", "he liked to play with totf", "the waitress was not amusd", "in hopes of finding the trufg", "fishermen never forget their polsdf", "i greatly apprecatwae", "the doll spun arndsa", "tomatoes do not belong on a hambrsfs", "he dreamed of leaving his hmag", "animal crackers and peanut butttra", "climb the ladder and fall downwrdsg", "she opened her third bottle of wn",
"waffles and pancakrs", "hello worls", "spelling erris", "red rabbut", "basketball and volelyball", "I arrived at the bus station early but waited until noon for the bys", "where are you frpn", "so the horrbily outnumbers mongouse",
"dude imagine being a bug and getting stuck in a motorcagl", "dogs only bark at their neighbasd", "consequences of my actaslkhf",
"in elementary school i used kgayons", "someone is under the mistlae", "I overheard someonaga", "i hate all bugs and beetlag"]
correctWord = ["letter", "paintings", "dense", "mirror", "lipstick", "corner", "information", "toys", "amused", "truth", "poles", "appreciate", "around", "hamburgers", "home", "butter", "downwards", "wine",
"pancakes", "world", "errors" ,"rabbit", "volleyball", "bus", "from", "mongoose", "motorcycle", "neighbors", "actions", "crayons",
"mistletoe", "someone", "beetles"]
#print("Naive Tests:")
#naiveTest(inputs)

models = [twitter_bigram_model, twitter_trigram_model, reuters_trigram_model]
#print("N-gram Tests:")
#nGramTest(inputs, models)

#Processed words per second
totalTime = 0
index = 0
correctWords = 0
for input in longInput:
    index += 1
    output, timeOutput = correct_sentence(input, twitter_terms, twitter_total_words, 1)
    totalTime += timeOutput
    wordOutput = (output[len(output) - 1])[1]
    if wordOutput == correctWord[index - 1]:
        correctWords += 1

print(correctWords/len(correctWord))
print(totalTime/index)