import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from log_file_2 import Set_levels
import pickle
import matplotlib.pyplot as plt
logger = Set_levels("final_file")

def f_m(final_X_train,final_y_train,final_X_test,final_y_test):
    try:
        # logistic regression
        log_reg = LogisticRegression()
        log_reg.fit(final_X_train, final_y_train)
        log_pred = log_reg.predict(final_X_test)
        with open('Breast-Cancer.pkl','wb') as f_:
            pickle.dump(log_reg,f_)
        logger.info("final Model saved Successfully")

    except Exception as e:
        er_type,er_msg,er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")