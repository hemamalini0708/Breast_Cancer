import numpy as np
import pandas as pd
import sys
import logging
from log_file_2 import Set_levels
logger = Set_levels("hypothesis_file")
from scipy.stats import pearsonr

def hypothesis(X_train, Y_train, X_test, Y_test):
    try:
        cor_personr_value = []
        waste_columns = []

        for i in X_train.columns:
            result = pearsonr(X_train[i], Y_train)
            cor_personr_value.append(result)

        cor_personr_value = np.array(cor_personr_value)
        p_value = pd.Series(cor_personr_value[:, 1], index=X_train.columns)

        for j in p_value.index:
            if p_value[j] > 0.05:
                waste_columns.append(j)
                logger.info(f"Targeted Column: {j} → p-value: {p_value[j]:.6f}")

        X_train = X_train.drop(waste_columns, axis=1)
        X_test = X_test.drop(waste_columns, axis=1)

        return X_train, X_test

    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")
