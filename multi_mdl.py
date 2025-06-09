import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sklearn
import sys
import logging
from log_file_2 import Set_levels
logger = Set_levels("models")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# finding Best K value for KNN
def KNN(X_train,Y_train,X_test,Y_test):
    try:
        values_accu = []
        k_values = np.arange(3,50,2)
        for i in k_values:
            kn_reg = KNeighborsClassifier(n_neighbors=i)
            kn_reg.fit(X_train,Y_train)
            values_accu.append(accuracy_score(Y_test, kn_reg.predict(X_test)))
        logger.info(f"All K values Accuracy : {values_accu}")
        logger.info(f"best K value : {k_values[values_accu.index(max(values_accu))]} with Accuracy : {max(values_accu)}")
        # After getting best k value we need to fit again
        kn_reg = KNeighborsClassifier(n_neighbors=i)
        kn_reg.fit(X_train, Y_train)
        logger.info(f"KNN Accuracy : {accuracy_score(Y_test, kn_reg.predict(X_test))}")
        logger.info(f"KNN Confusion Matrix : {confusion_matrix(Y_test, kn_reg.predict(X_test))}")
        logger.info(f"KNN Classification Report : {classification_report(Y_test, kn_reg.predict(X_test))}")
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def NB(X_train,Y_train,X_test,Y_test):
    try:
        NB_reg = GaussianNB()
        NB_reg.fit(X_train,Y_train)
        logger.info(f"NB Accuray : {accuracy_score(Y_test, NB_reg.predict(X_test))}")
        logger.info(f"NB Confusion Matrix : {confusion_matrix(Y_test, NB_reg.predict(X_test))}")
        logger.info(f"NB Classification Report : {classification_report(Y_test, NB_reg.predict(X_test))}")
    except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def LR(X_train,Y_train,X_test,Y_test):
    try:
        LR_reg = LogisticRegression()
        LR_reg .fit(X_train,Y_train)
        logger.info(f"LR Accuray : {accuracy_score(Y_test, LR_reg .predict(X_test))}")
        logger.info(f"LR Confusion Matrix : {confusion_matrix(Y_test, LR_reg .predict(X_test))}")
        logger.info(f"LR Classification Report : {classification_report(Y_test, LR_reg .predict(X_test))}")
    except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def DT(X_train,Y_train,X_test,Y_test):
    try:
        DT_reg = DecisionTreeClassifier(criterion='entropy')
        DT_reg.fit(X_train,Y_train)
        logger.info(f"DT Accuray : {accuracy_score(Y_test,  DT_reg.predict(X_test))}")
        logger.info(f"DT Confusion Matrix : {confusion_matrix(Y_test,  DT_reg.predict(X_test))}")
        logger.info(f"DT Classification Report : {classification_report(Y_test,  DT_reg.predict(X_test))}")
    except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def RF(X_train,Y_train,X_test,Y_test):
    try:
        value_accu = []
        trees = np.random.randint(1, 100, 10)  # start from 1 to avoid zero
        for j in trees:
            RF_reg = RandomForestClassifier(criterion='entropy', n_estimators=j)
            RF_reg.fit(X_train,Y_train)
            value_accu.append(accuracy_score(Y_test, RF_reg.predict(X_test)))
        logger.info(f"Accuracy for all tree values : {value_accu}")
        best_tree = trees[value_accu.index(max(value_accu))]
        logger.info(f"Best Tree value : {best_tree} with Accuracy {max(value_accu)}")
        # Fit with best tree value
        RF_reg = RandomForestClassifier(criterion='entropy', n_estimators=best_tree)
        RF_reg.fit(X_train, Y_train)
        logger.info(f"RF Accuracy : {accuracy_score(Y_test, RF_reg.predict(X_test))}")
        logger.info(f"RF Confusion Matrix : {confusion_matrix(Y_test, RF_reg.predict(X_test))}")
        logger.info(f"RF Classification Report : {classification_report(Y_test, RF_reg.predict(X_test))}")
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")


def multi_models(X_train,Y_train,X_test,Y_test):
    try:
        logger.info(f"------K-Nearst-Neighbor-------")
        KNN(X_train, Y_train, X_test, Y_test)
        logger.info(f"---------Navi_Bayes-----------")
        NB(X_train, Y_train, X_test, Y_test)
        logger.info(f"------Logistic Regression------")
        LR(X_train, Y_train, X_test, Y_test)
        logger.info(f"--------Decision Tree---------")
        DT(X_train, Y_train, X_test, Y_test)
        logger.info(f"--------Random Forest---------")
        RF(X_train, Y_train, X_test, Y_test)
    except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")