import numpy as np

def cosine_similarity(A, B):
    """
    :param A: a numpy array which corresponds to a word vector
    :param B: a numpy array which corresponds to a word vector
    :return cos: numerical number representing the cosine similarity between A and B
    """
    # set variable to the true label
    cos = -10

    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot / (norma * normb)

    return cos

def nearest_neighbor(v, candidates, k=1):
    """
    :param v: the vector that are going to fined nearest neighbor
    :param candidates: a set of vectors where we will find the neighbors
    :param k: top k nearest neighbors to find
    :return k_idx: the indices ot the top k closet vectors in sorted form
    """
    similarity_l = []

    for row in candidates:
        # get the cosine similarity
        cos_similarity = cosine_similarity(v, row)

        # append the similarity to the list
        similarity_l.append(cos_similarity)

    # sort the similarity list and get the indices of the sorted list
    sorted_ids = np.argsort(similarity_l)

    # get the indices of the k most similar candidate vectors
    k_idx = sorted_ids[-k:]

    return  k_idx

def test_vocabulary(X, Y, R):
    """
    :param X: a matrix where the columns are the english embeddings
    :param Y: a matrix where the columns corresponds to the french embeddings
    :param R: the transform matrix which translates word embedding form
    :return accuracy: for the english to french capitals
    """
    # prediction
    pred = np.dot(X, R)

    # initialize the number correct
    num_correct = 0

    # loop through each row in pred
    for i in range(len(pred)):
        # get the idx of the nearest neighbor of pred at row 'i'
        pred_idx = nearest_neighbor(pred[i], Y, k=1)

        # if the idx of the nearest neighbor equals the row of i
        if pred_idx == i:
            # increment the number correct by 1
            num_correct += 1

    # accuracy = number correct / number of rows in pred
    accuracy = num_correct / len(pred)

    return accuracy













