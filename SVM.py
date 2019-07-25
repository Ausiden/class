import numpy as np
def svm_loss_native(w, x_train, y_train, reg):  # reg正则系数
    num_classes = w.shape[1]
    num_train = x_train.shape[0]
    dw = np.zeros(w.shape)
    loss = 0
    for i in range(num_train):
        score = x_train[i].dot(w)
        correct_score = score[y_train[i]]
        for j in range(num_classes):
            if j == y_train[i]:
                continue
            margin = max(0, score[j] - correct_score + 1)
            if margin > 0:
                loss += margin
                dw[:, j] += x_train[i].T
                dw[:, y_train[i]] += -x_train[i].T
    loss /= num_train
    dw /= num_train
    loss += 0.5 * reg * np.sum(w * w)
    dw += reg * w
    return loss, dw


def svm_loss_vectorized(w, x_train, y_train, reg):
    num_train = x_train.shape[0]
    num_classes = w.shape[1]
    loss = 0
    score = x_train.dot(w)
    correct_score = score[range(num_train), y_train].reshape(-1, 1)
    margin = np.maximum(0, score - correct_score + 1)
    margin[range(num_train), y_train] = 0
    loss += np.sum(margin) / num_train + 0.5 * reg * np.sum(w * w)
    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margin > 0] = 1
    coeff_mat[range(num_train), y_train] = -np.sum(coeff_mat, axis=1)
    dw = x_train.T.dot(coeff_mat)
    dw = dw / num_train + reg * w
    return loss, dw



