import pickle

import pandas as pd
from gensim.models import KeyedVectors

def get_dict(file_name):
    """
    Returns the english to french dictionary given a file where the each column corresponds to a word
    """
    my_file = pd.read_csv(file_name, delimiter=' ')

    # the english to french dictionary to be returned
    etof = {}

    for i in range(len(my_file)):

        # indexing into the rows.
        en = my_file.loc[i][0]
        fr = my_file.loc[i][1]
        etof[en] = fr

    return etof

en_embeddings = KeyedVectors.load_word2vec_format('../input/GoogleNews-vectors-negative300.bin', binary = True)
fr_embeddings = KeyedVectors.load_word2vec_format('../input//wiki.multi.fr.vec')


# loading the english to french dictionaries
en_fr_train = get_dict('../input/en-fr.train.txt')
print('The length of the english to french training dictionary is', len(en_fr_train))
en_fr_test = get_dict('../input/en-fr.test.txt')
print('The length of the english to french test dictionary is', len(en_fr_train))

english_set = set(en_embeddings.vocab)
french_set = set(fr_embeddings.vocab)
en_embeddings_subset = {}
fr_embeddings_subset = {}
french_words = set(en_fr_train.values())

for en_word in en_fr_train.keys():
    fr_word = en_fr_train[en_word]
    if fr_word in french_set and en_word in english_set:
        en_embeddings_subset[en_word] = en_embeddings[en_word]
        fr_embeddings_subset[fr_word] = fr_embeddings[fr_word]


for en_word in en_fr_test.keys():
    fr_word = en_fr_test[en_word]
    if fr_word in french_set and en_word in english_set:
        en_embeddings_subset[en_word] = en_embeddings[en_word]
        fr_embeddings_subset[fr_word] = fr_embeddings[fr_word]


pickle.dump( en_embeddings_subset, open("../input/en_embeddings.p", "wb" ) )
pickle.dump( fr_embeddings_subset, open("../input/fr_embeddings.p", "wb" ) )