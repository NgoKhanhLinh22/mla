from utils import *
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_likelihood = 0.
    for idx, u_id in enumerate(data["user_id"]):
        q_id = data["question_id"][idx]
        log_likelihood+= (
                    data["is_correct"][idx] * (theta[u_id] - beta[q_id]) - np.log1p(np.exp(theta[u_id] - beta[q_id])))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_likelihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id_arr = np.array(data["user_id"])
    question_id_arr = np.array(data["question_id"])
    correct_id_arr = np.array(data["is_correct"])

    theta_copy = theta.copy()
    beta_copy = beta.copy()
    for i in range(len(theta)):
        theta[i] -= lr * (np.sum(sigmoid(theta_copy[i] - beta_copy)[question_id_arr[user_id_arr == i]]) - np.sum(correct_id_arr[user_id_arr == i]))

    for j in range(len(beta)):
        beta[j] -= lr * (np.sum(correct_id_arr[question_id_arr == j]) - np.sum(sigmoid(theta_copy - beta_copy[j])[user_id_arr[question_id_arr == j]]))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(train_data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    np.random.seed(1)
    theta = np.random.rand(542) * 0.1
    beta = np.random.rand(1774) * 0.1

    val_acc_lst, train_acc_lst = [], []
    val_log_likelihood, train_log_likelihood = [], []

    for i in range(iterations):
        # Log likelihood
        train_neg_lld = neg_log_likelihood(train_data, theta=theta, beta=beta)
        train_log_likelihood.append(train_neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_log_likelihood.append(val_neg_lld)

        train_score = evaluate(data=train_data, theta=theta, beta=beta)
        train_acc_lst.append(train_score)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(val_score)
        #
        print("NLLK: {} \t Train Score: {} \t Validation Score: {}".format(train_neg_lld, train_score, val_score))
        theta, beta = update_theta_beta(train_data, lr, theta, beta)


    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_log_likelihood, train_log_likelihood


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

def create_sparse_matrix(data):
    matrix = np.empty(shape=(542, 1774))
    matrix[:] = np.NaN
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        mark = data["is_correct"][i]
        matrix[u, q] = mark
    return matrix


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr, iterations = 0.01, 20

    theta, beta, val_log_likelihood, train_log_likelihood = \
        irt(train_data, val_data, lr, iterations)

    plt.plot(train_log_likelihood, label="train");
    plt.plot(val_log_likelihood, label="valid");
    plt.ylabel("Negative Log Likelihood");
    plt.xlabel("Iteration");
    plt.xticks(np.arange(0, iterations, 1));
    # plt.yticks(np.logspace(3, 5, 20));
    plt.title("Neg Log Likelihood for Train and Validation Data");
    plt.legend();
    plt.savefig("../my_photo/Neg Log Likelihood for Train and Validation Data.png");
    plt.show();
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    val_score = evaluate(data=val_data, theta=theta, beta=beta)
    test_score = evaluate(data=test_data, theta=theta, beta=beta)

    np.save("theta.npy", theta)
    np.save("beta.npy", beta)
    np.save("val_nllk.npy", val_log_likelihood)
    np.save("train_nllk.npy", train_log_likelihood)


    print("Validation Accuracy: ", val_score)
    print("Test Accuracy: ", test_score)

    # Part D
    selected_questions = np.array([100, 200, 300, 400, 500])
    theta = theta.reshape(-1)
    theta.sort()
    for selected in selected_questions:
        plt.plot(theta, sigmoid(theta - beta[selected]), label=f"Question {selected}");
    plt.ylabel("Probability");
    plt.xlabel("Theta");
    plt.title("Selected Questions Probability as a function of Theta");
    plt.legend();
    plt.savefig("../my_photo/Selected Questions Probability as a function of Theta.png");
    plt.show();


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
