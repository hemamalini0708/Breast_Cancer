import numpy as np
import pandas as pd
import sklearn
import sys
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from log_file_2 import Set_levels
logger = Set_levels("best_model")


def selecting_best_model(X_train, Y_train, X_test, Y_test):
    try:
        knn_reg = KNeighborsClassifier(n_neighbors=3)
        knn_reg.fit(X_train, Y_train)
        knn_probs = knn_reg.predict_proba(X_test)[:, 1]

        nb_reg = GaussianNB()
        nb_reg.fit(X_train, Y_train)
        nb_probs = nb_reg.predict_proba(X_test)[:, 1]

        LR_reg = LogisticRegression()
        LR_reg.fit(X_train, Y_train)
        LR_probs = LR_reg.predict_proba(X_test)[:, 1]

        DT_reg = DecisionTreeClassifier(criterion='entropy')
        DT_reg.fit(X_train, Y_train)
        DT_probs = DT_reg.predict_proba(X_test)[:, 1]

        RF_reg = RandomForestClassifier(n_estimators=99, criterion='entropy')
        RF_reg.fit(X_train, Y_train)
        RF_probs = RF_reg.predict_proba(X_test)[:, 1]

        fpr_knn, tpr_knn, _ = roc_curve(Y_test, knn_probs)
        fpr_nb, tpr_nb, _ = roc_curve(Y_test, nb_probs)
        fpr_LR, tpr_LR, _ = roc_curve(Y_test, LR_probs)
        fpr_DT, tpr_DT, _ = roc_curve(Y_test, DT_probs)
        fpr_RF, tpr_RF, _ = roc_curve(Y_test, RF_probs)

        plt.figure(figsize=(6, 4))
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.plot(fpr_knn, tpr_knn, color='r', label="KNN")
        plt.plot(fpr_nb, tpr_nb, color='b', label="NB")
        plt.plot(fpr_LR, tpr_LR, color='g', label="LR")
        plt.plot(fpr_DT, tpr_DT, color='black', label="DT")
        plt.plot(fpr_RF, tpr_RF, color='y', label="RF")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend(loc="lower right")
        plt.show()

    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")
