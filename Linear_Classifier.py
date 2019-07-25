import numpy as np
import SVM
import Softmax


class LinearClassifier:
    def __init__(self):
        self.w = None

    def train(self, xtrain, ytrain, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train, dim = xtrain.shape
        num_classes = np.max(ytrain) + 1
        if self.w is None:
            self.w = 0.001 * np.random.rand(dim, num_classes)
        loss_history = []
        for it in range(num_iters):
            x_batch = None
            y_batch = None
            batch_idx = np.random.choice(num_train, batch_size, replace=True)
            x_batch = xtrain[batch_idx]
            y_batch = ytrain[batch_idx]
            loss, grad = self.loss(x_batch, y_batch, reg)
            loss_history.append(loss)
            self.w += -1 * learning_rate * grad
            if verbose and it % 100 == 0:
                print("iteration %d/%d:loss %f" % (it, num_iters, loss))
        return loss_history

    def predict(self, xtrain):
        y_pred = np.zeros(xtrain.shape[1])
        scores = xtrain.dot(self.w)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def loss(self, x_batch, y_batch, reg):
        pass


class LinearSVM(LinearClassifier):
    def loss(self, x_batch, y_batch, reg):
        return SVM.svm_loss_vectorized(self.w, x_batch, y_batch, reg)


class SoftMax(LinearClassifier):
    def loss(self, x_batch, y_batch, reg):
        return Softmax.softmax_loss_vectorized(self.w, x_batch, y_batch, reg)
