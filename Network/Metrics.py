import numpy as np
import sklearn.metrics as metrics


def numeric_score(pred, gt, thre=0.5, is_tensor=True,):
    if is_tensor:
        pred = pred.view(-1).detach().to('cpu').numpy()
        gt = gt.view(-1).detach().to('cpu').numpy()

    auc_score = 0.0
    try:
        auc_score = metrics.roc_auc_score(gt, pred)
    except ValueError:
        pass

    pred[pred <= thre] = 0.0
    pred[pred > thre] = 1.0

    tn, fp, fn, tp = metrics.confusion_matrix(gt, pred, labels=[0, 1]).ravel()

    acc = (tp + tn) / (tp + fp + tn + fn + 1e-5)
    sen = (tp + 1e-5) / (tp + fn + 1e-5)
    pre = (tp + 1e-5) / (tp + fp + 1e-5)
    spe = (tn + 1e-5) / (tn + fp + 1e-5)
    balanced_acc = (sen+spe) / 2.0
    return balanced_acc, acc, auc_score, pre, sen, spe


def dice_score(pred, gt, thre=0.5, is_tensor=True):
    if is_tensor:
        pred = pred.detach().to('cpu').numpy()
        gt = gt.detach().to('cpu').numpy()
        pred = pred.reshape((pred.shape[0], -1))
        gt = gt.reshape((pred.shape[0], -1))
    pred[pred < thre] = 0.0
    pred[pred >= thre] = 1.0

    num = 2 * np.sum(pred * gt, axis=1)
    den = np.sum(pred + gt, axis=1) + 1e-5
    dice = np.mean(num / den)
    return dice