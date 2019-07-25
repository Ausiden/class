import numpy as np


class twolayernet():
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['w1'] = std * np.random.rand(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = std * np.random.rand(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, x, y=None, reg=0.0):
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        num_train = x.shape[0]
        h_output = np.maximum(0, x.dot(w1) + b1)
        score = h_output.dot(w2) + b2
        if y is None:
            return score
        loss = None
        shift_score = score - np.max(score, axis=1).reshape(-1, 1)
        softmax_output = np.exp(shift_score) / np.sum(np.exp(shift_score), axis=1).reshape(-1, 1)
        loss = -np.sum(np.log(softmax_output[range(num_train), y]))
        loss /= num_train
        loss += reg * 0.5 * (np.sum(w1 * w1) + np.sum(w2 * w2))  # 正则项
        grads = {}

        # 第二梯度的计算
        dscore = softmax_output.copy()
        dscore[range(num_train), y] -= 1
        dscore /= num_train
        grads['w2'] = h_output.T.dot(dscore) + reg * w2
        grads['b2'] = np.sum(dscore, axis=0)

        # 第一梯度的计算
        dh = dscore.dot(w2.T)
        dh_ReLu = (h_output > 0) * dh
        grads['w1'] = x.T.dot(dh_ReLu) + reg * w1
        grads['b1'] = np.sum(dh_ReLu, axis=0)
        return loss, grads

    def train(self, x, y, x_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        num_train = x.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)  # 每一轮迭代次数
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        for it in range(num_iters):
            x_batch = None
            y_batch = None
            idx = np.random.choice(num_train, batch_size, replace=True)
            x_batch = x[idx]
            y_batch = y[idx]
            loss, grads = self.loss(x_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # 参数更新
            self.params['w2'] += -learning_rate * grads['w2']
            self.params['b2'] += -learning_rate * grads['b2']
            self.params['w1'] += -learning_rate * grads['w1']
            self.params['b1'] += -learning_rate * grads['b1']

            if verbose and it % 100 == 0:  # 每迭代100次，打印
                print("iteration %d/%d:loss %f" % (it, num_iters, loss))
            if it % iterations_per_epoch == 0:  # 一轮迭代结束
                train_acc = (self.predict(x_batch) == y_batch).mean()
                val_acc = (self.predict(x_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # 更新学习率
                learning_rate *= learning_rate_decay
        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }

    def predict(self, x):
        y_pred = None
        h = np.maximum(0, x.dot(self.params['w1']) + self.params['b1'])
        scores = h.dot(self.params['w2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        return y_pred
