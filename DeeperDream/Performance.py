import numpy as np
from matplotlib import pyplot as plt


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
    return 1 - top /bot