import numpy as np
from matplotlib import pyplot as plt


def Sigmoid(z):
    return 1 / (1+np.exp(-z))

def SoftMax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

## Loss Functions

def BinaryCrossEntropy(y, p_hat):
    return -1 * np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))

def MSE(y, y_hat):
    return np.trace((y - y_hat).T @ (y - y_hat)) / y.shape[0]

def CrossEntropy(y, p_hat):
    """
    One-Hot Encoded
    """
    return -np.sum(y * np.log(p_hat))
    
    
## Performance Metrics
def TruePositive(y, y_hat):
    return np.sum(y &  y_hat)
    
def FalsePositive(y, y_hat):
    return np.sum(~y &  y_hat)
    
def TrueNegative(y, y_hat):
    return np.sum(~y & ~y_hat)
    
def FalseNegative(y, y_hat):
    return np.sum(y & ~y_hat)

def Accuracy(y, p_hat, t=0.5):
    y_hat = p_hat >= t
    y = y == 1
    tp = TruePositive(y, y_hat)
    tn = TrueNegative(y, y_hat)
    return (tp + tn) / y.shape[0]

def Precision(y, p_hat, t=0.5):
    y_hat = p_hat >= t
    y = y == 1
    tp = TruePositive(y, y_hat)
    fp = FalsePositive(y, y_hat)
    return tp / (tp + fp)
    
def Recall(y, p_hat, t=0.5):
    y_hat = p_hat >= t
    y = y == 1
    tp = TruePositive(y, y_hat)
    fn = FalseNegative(y, y_hat)
    return tp / (tp + fn)

def F1(y, p_hat, t=0.5):
    precision = Precision(y, p_hat, t)
    recall = Recall(y, p_hat, t)
    f1 = 2 * precision * recall / (precision + recall)
    if np.isnan(f1):
        return 0
    else:
        return f1
    
def TPR(y, p_hat, t=0.5):
    y_hat = p_hat >= t
    y = y == 1
    tp = TruePositive(y, y_hat)
    fn = FalseNegative(y, y_hat)
    return tp / (tp + fn)

def FPR(y, p_hat, t=0.5):
    y_hat = p_hat >= t
    y = y == 1
    fp = FalsePositive(y, y_hat)
    tn = TrueNegative(y, y_hat)
#     print(fp)
    return fp / (fp + tn)

def ROC(y, p_hat, num = 1000):
    tpr = []
    fpr = []
    f1 = []
    accuracy = []
    
    thresholds = np.linspace(1, 0, num=num)
    for t in thresholds:
#         print(t)
        tpr.append(TPR(y, p_hat, t))
        fpr.append(FPR(y, p_hat, t))
        f1.append(F1(y, p_hat, t))
        accuracy.append(Accuracy(y, p_hat, t))
    
    opt_f1 = np.argmax(f1)
    opt_acc = np.argmax(accuracy)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k:')
    plt.scatter(fpr[opt_f1], [tpr[opt_f1]], c='g')
    plt.scatter(fpr[opt_acc], [tpr[opt_acc]], c='r')
    plt.title("ROC Curve")
    
    tpr = np.array(tpr)
    AUC = np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2 )
    
    print("Best F1 Score:         {:.3} at {:.3}".format(f1[opt_f1], thresholds[opt_f1]))
    print("Best Accuracy Score:   {:.3} at {:.3}".format(f1[opt_acc], thresholds[opt_acc]))
    print("Area Under the Curve:  {:.3}".format(AUC))
    
    return AUC

def R2(y, y_hat):
    y_bar = np.mean(y)
    top = np.sum((y - y_hat) ** 2)
    bot = np.sum((y - y_bar) ** 2)
    return 1 - top/bot


## Preprocessing
def FlatNormalize(x, cols: list = []):
    """
    Change the range features of data to [0, 1]
    This will normalize in place
    
    :param x: The data to be normalized - expects each column to be a feature and each row to be a a data point.
    :param cols: List of indices of columns you want to normalize.
    """
    if not cols:
        cols = list(range(x.shape[1]))
    x[:, cols] -= np.min(x[:, cols], axis=0)
    x[:, cols] /= np.max(x[:, cols], axis=0)
    return x


## Models

class Linear_Regression:
    def __init__(self, num_iter=1000, tol=1e-4, learning_rate=1e-4, alpha=0, beta=0.5):
        self.num_iter = num_iter
        self.tolerance = tol
        self.learning_rate = learning_rate
        self.w = 0
        self.alpha = alpha
        self.beta = beta

    def Fit(self, x, y, pad=False, plot_loss = False):
        """
        This method get x and y nd arrays and apply the gradient descent method.

        :param x: nd array
        :param y: nd array
        :param pad: boolean arguement to add y-intercept

        :return: self.beta
        """
        loss = [np.inf]

        if pad:
            x = np.hstack([np.ones((x.shape[0], 1)), x])

        self.w = np.random.randn(x.shape[1], y.shape[1])

        for i in range(self.num_iter):
            y_hat = x @ self.w
            loss.append(np.trace((y - y_hat).T @ (y - y_hat)/y.shape[0]))
            grad = -x.T @ (y - y_hat) + \
                self.alpha * (
                    self.beta * np.sign(self.w) + # L1
                    (1 - self.beta) * self.w      # L2
                )
            self.w -= self.learning_rate * grad
            if abs(loss[-1] - loss[-2]) < self.tolerance:
                break
        if plot_loss:
            plt.plot(loss)
        return self.w
    
    def Predict(self, x, pad=False):
        if pad:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
        y_hat = x @ self.w
        return y_hat

    
class Binary_Logistic_Regression:
    def __init__(self, num_iter=1000, tol=1e-4, learning_rate=1e-4, alpha=0, beta=0.5):
        self.num_iter = num_iter
        self.tolerance = tol
        self.learning_rate = learning_rate
        self.w = 0
        self.alpha = alpha
        self.beta = beta

    def Fit(self, x, y, pad=False, plot_loss = False):
        """
        This method get x and y nd arrays and apply the gradient descent method.

        :param x: nd array
        :param y: nd array
        :param pad: boolean arguement to add y-intercept

        :return: self.beta
        """
        loss = [np.inf]

        if pad:
            x = np.hstack([np.ones((x.shape[0], 1)), x])

        self.w = np.random.randn(x.shape[1], y.shape[1])

        for i in range(self.num_iter):
            p_hat = self.Predict(x, pad=False)
            loss.append(BinaryCrossEntropy(y, p_hat))
            grad = -x.T @ (y - p_hat) + \
                self.alpha * (
                    self.beta * np.sign(self.w) + # L1
                    (1 - self.beta) * self.w      # L2
                )
            self.w -= self.learning_rate * grad
            if abs(loss[-1] - loss[-2]) < self.tolerance:
                break
        if plot_loss:
            plt.plot(loss)
        return self.w
    
    def Predict(self, x, pad=False):
        if pad:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
        y_hat = x @ self.w
        p_hat = Sigmoid(y_hat)
        return p_hat


class Logistic_Regression:
    def __init__(self, num_iter=1000, tol=1e-4, learning_rate=1e-4, alpha=0, beta=0.5):
        self.num_iter = num_iter
        self.tolerance = tol
        self.learning_rate = learning_rate
        self.w = 0
        self.alpha = alpha
        self.beta = beta

    def Fit(self, x, y, pad=False, plot_loss = False):
        """
        This method get x and y nd arrays and apply the gradient descent method.

        :param x: nd array
        :param y: nd array
        :param pad: boolean arguement to add y-intercept

        :return: self.beta
        """
        loss = [np.inf]

        if pad:
            x = np.hstack([np.ones((x.shape[0], 1)), x])

        self.w = np.random.randn(x.shape[1], y.shape[1])

        for i in range(self.num_iter):
            p_hat = self.Predict(x, pad=False)
            loss.append(CrossEntropy(y, p_hat))
            grad = x.T @ (p_hat - y) + \
                self.alpha * (
                    self.beta * np.sign(self.w) + # L1
                    (1 - self.beta) * self.w      # L2
                )
            self.w -= self.learning_rate * grad
            if abs(loss[-1] - loss[-2]) < self.tolerance:
                break
        if plot_loss:
            plt.plot(loss)
        return self.w
    
    def Predict(self, x, pad=False):
        if pad:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
        y_hat = x @ self.w
        p_hat = SoftMax(y_hat)
        return p_hat
