import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize


# ========================
# 基础指标
# ========================

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def multiclass_auc(y_true, y_prob, num_classes):
    """
    多分类 AUC（OVR方式，论文常用）
    """
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    return roc_auc_score(
        y_true_bin,
        np.array(y_prob),
        average="macro",
        multi_class="ovr"
    )


# ========================
# ROC 曲线数据
# ========================

def compute_roc_data(y_true, y_prob, num_classes):
    """
    返回每个类别的 ROC 曲线数据
    """
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(y_prob)[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], np.array(y_prob)[:, i])

    return fpr, tpr, roc_auc


# ========================
# PR 曲线数据
# ========================

def compute_pr_data(y_true, y_prob, num_classes):
    """
    返回每个类别的 PR 曲线数据
    """
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    precision = dict()
    recall = dict()

    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i],
            np.array(y_prob)[:, i]
        )

    return precision, recall


# ========================
# 混淆矩阵
# ========================

def compute_confusion_matrix(y_true, y_pred):
    """
    返回 confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


# ========================
# 混淆矩阵绘图（论文级）
# ========================

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    生成混淆矩阵图（可保存）
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()
