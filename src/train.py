import pickle

import feature_generator
import models
import predict
import word_embeddings

if __name__ == "__main__":
    # load english and french embeddings
    en_embeddings_subset = pickle.load(open("../input/en_embeddings.p", "rb"))
    fr_embeddings_subset = pickle.load(open("../input/fr_embeddings.p", "rb"))

    # load the english to french dictionary
    en_fr_train = word_embeddings.get_dict('../input/en-fr.train.txt')
    en_fr_test = word_embeddings.get_dict('../input/en-fr.test.txt')

    # get the train set
    X_train, Y_train = feature_generator.get_matrics(
        en_fr_train,
        fr_embeddings_subset,
        en_embeddings_subset
    )

    # calculate transformation matrix R
    R_train = models.align_embeddings(
        X_train,
        Y_train,
        train_steps=400,
        learning_rate=0.8
    )

    # get the valid set
    X_val, Y_val = feature_generator.get_matrics(
        en_fr_test,
        fr_embeddings_subset,
        en_embeddings_subset
    )

    # calculate and print out the accuracy
    acc = predict.test_vocabulary(X_val, Y_val, R_train)
    print(f"accuracy on test set is {acc:.3f}")


