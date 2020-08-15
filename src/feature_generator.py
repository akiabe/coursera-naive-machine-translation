import numpy as np

def get_matrics(en_fr, french_vecs, english_vecs):
    """
    :param en_fr: english to french dictionary
    :param french_vecs: french words to their corresponding word embeddings
    :param english_vecs: english words to their corresponding word embeddings
    :return X: a matrix where the columns are the english embeddings
    :return Y: a matrix where the columns correspond to the french embeddings
    :return R: the projection matrix that minimizes the F norm |X R -Y||^2
    """
    # create english and french word embeddings list
    X_l = list()
    Y_l = list()

    # get english and french words and store in a set()
    english_set = set(english_vecs.keys())
    french_set = set(french_vecs.keys())

    # loop through all english, french word pairs in the english french dictionary
    for en_word, fr_word in en_fr.items():

        # check that the french word has an embedding and that the english word has an embedding
        if fr_word in french_set and en_word in english_set:

            # get english and french embeddings
            en_vec = english_vecs[en_word]
            fr_vec = french_vecs[fr_word]

            # add the english and french embedding to the list
            X_l.append(en_vec)
            Y_l.append(fr_vec)

    # stack the vectors of X_l and Y_l into each matrix
    X = np.vstack(X_l)
    Y = np.vstack(Y_l)

    return X, Y