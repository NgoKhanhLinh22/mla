from fileinput import filename

from sklearn.impute import KNNImputer
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy User-based with k = {} : {}".format(k, acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy Item_based with k = {} : {}".format(k, acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    ######################################`###############################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_set = [1, 6, 11, 16, 21, 26]
    accuracy_user = []
    accuracy_item = []
    for k in k_set:
        accuracy_user.append(knn_impute_by_user(sparse_matrix, val_data, k))
        best_k_user = k_set[accuracy_user.index(max(accuracy_user))]

    for k in k_set:
        accuracy_item.append(knn_impute_by_item(sparse_matrix, val_data, k))
        best_k_item = k_set[accuracy_item.index(max(accuracy_item))]

    test_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    test_item = knn_impute_by_item(sparse_matrix, test_data, best_k_item)

    print("Best k for user is", best_k_user, "with test acc:", test_user)
    print("Best k for item is", best_k_item, "with test acc:", test_item)

    title = 'Accuracy on Validation Data User-based'
    acc = accuracy_user
    for i in range(2):
        # Generate the plot for validation accuracy.
        if (i == 1):
            title = 'Accuracy on Validation Data Item-based'
            acc = accuracy_item
        plt.figure()
        plt.title(title)
        plt.plot(k_set, acc, label="valiadtion")
        plt.xlabel("k")
        plt.xticks(k_set)
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig("../my_photo/" + title+ ".png")
        plt.show()

        #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()