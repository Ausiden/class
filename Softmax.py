import numpy as np


def softmax_loss_native(w, x_train, y_train, reg):  # reg正则系数
    num_classes = w.shape[1]
    num_train = x_train.shape[0]
    dw = np.zeros(w.shape)
    loss = 0
    for i in range(num_train):
        scores = x_train[i].dot(w)
        shift_scores = scores - max(scores)  #max函数防止增长过快，呈指数型增长
        loss_i = -shift_scores[y_train[i]] + np.log(sum(np.exp((shift_scores))))
        loss += loss_i
        for j in range(num_classes):
            softmax_out = np.exp(shift_scores[j]) / sum(np.exp(shift_scores))
            if j == y_train[i]:
                dw[:, j] += (-1 + softmax_out) * x_train[i]
            else:
                dw[:, j] += softmax_out * x_train[i]
    loss /= num_train
    dw /= num_train
    loss += 0.5 * reg * np.sum(w * w)
    dw += reg * w
    return loss, dw


def softmax_loss_vectorized(w, x_train, y_train, reg):
    num_train = x_train.shape[0]
    num_classes = w.shape[1]
    loss = 0
    score = x_train.dot(w)
    shift_score = score - np.max(score, axis=1).reshape(-1, 1)
    softmax_output = np.exp(shift_score) / np.sum(np.exp(shift_score), axis=1).reshape((-1, 1))
    loss = -np.sum(np.log(softmax_output[range(num_train), y_train]))
    loss /= num_train
    loss += 0.5 * reg * np.sum(w * w)
    dS = softmax_output.copy()
    dS[range(num_train), y_train] += -1
    dw = (x_train.T).dot(dS)
    dw = dw / num_train + reg * w
    return loss, dw
