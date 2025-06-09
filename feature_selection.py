import numpy as np
import pandas as pd
import sys
import logging
from log_file_2 import Set_levels
from sklearn.feature_selection import VarianceThreshold
logger = Set_levels("Feature_Selection_Area")
quansi_con = VarianceThreshold(threshold=0.1)

def constant(train_data, test_data):
    try:
        quansi_con.fit(train_data)
        dropped_cols = train_data.columns[~quansi_con.get_support()]
        logger.info(f"Low variance features dropped: {list(dropped_cols)}")
        logger.info("Low variance features removed successfully")
        return train_data.loc[:, quansi_con.get_support()], test_data.loc[:, quansi_con.get_support()]
    except Exception as e:
        logger.error(f"Low variance filter failed: {e}", exc_info=True)


