import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import sklearn
import pickle
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from log_file_2 import Set_levels
logger = Set_levels('main_1')
from sklearn.feature_selection import VarianceThreshold
quansi_con = VarianceThreshold(threshold=0.1)
from feature_selection import constant
from hypothesis_file import hypothesis
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from multi_mdl import multi_models
from best_model import  selecting_best_model
from finall_file import f_m
from finall_testing import testing_


class BREAST_CANCER:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info("Data loaded successfully")

            #checking null values
            self.col_with_nullvalues = []
            for i in self.df.columns:
                if self.df[i].isnull().sum() > 0:
                    self.col_with_nullvalues.append(i)
            logger.warning(f"Columns with null values : {self.col_with_nullvalues}")

            # Drop unwanted columns
            self.df = self.df.drop(['id'], axis=1)
            logger.info("Unwanted columns removed successfully")

            # X , Y
            self.X = self.df.iloc[:, 1:]
            self.Y = self.df.iloc[:, 0]

            # Train test split
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                self.X, self.Y, test_size=0.2, random_state=42)
            logger.info(f"Training data: {len(self.X_train), len(self.Y_train)}")
            logger.info(f"Testing data: {len(self.X_test), len(self.Y_test)}")

        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

    def variable_tf(self, train_num, test_num):
        try:
            train_num = np.log1p(train_num)
            test_num = np.log1p(test_num)
            logger.info("All numerical columns log-transformed successfully.")
            return train_num, test_num
        except Exception as e:
            logger.error(f"Log transform failed: {e}", exc_info=True)

    def outlier_handling(self, train_num, test_num):
        try:
            for col in train_num.columns:
                upper = train_num[col].quantile(0.95)
                lower = train_num[col].quantile(0.05)
                train_num[col] = np.clip(train_num[col], lower, upper)
                test_num[col] = np.clip(test_num[col], lower, upper)
            logger.info("Outliers handled using 5th–95th percentile clipping.")
            return train_num, test_num
        except Exception as e:
            logger.error(f"Outlier handling failed: {e}", exc_info=True)


    def cat_num(self):
        try:
            # Removing Numberical Columns
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            # Keeping Categorical Columns
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')
            logger.info(f"Catgeorical column from the data : {self.X_train_cat.columns}")
            logger.info(f"Categorical Columns from the data  : {self.X_test_cat.columns}")

            self.X_train_num,self.X_test_num = self.variable_tf(self.X_train_num,self.X_test_num)
            self.X_train_num,self.X_test_num = self.outlier_handling(self.X_train_num,self.X_test_num)
            logger.info(f"Outliers handled from the training and testing data successfully")
            logger.info(f"Feature Engineering Completed ")

            # Know we have to Combine Both Categorical & Numerical Columns by using Concatination
            self.training_ind_data = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.testing_ind_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)
            logger.info(f"Concatinated Both columns  Successfully")

            # To know the size of training_data testing_data
            logger.info(f"Size Of the Traning Data : {self.training_ind_data.shape}")
            logger.info(f"Size Of The Testing Data : {self.testing_ind_data.shape}")


            self.training_ind_data, self.testing_ind_data = constant(self.training_ind_data, self.testing_ind_data)
            logger.info(f"After Removing Low Variance Columns, shapes are: {self.training_ind_data.shape}, {self.testing_ind_data.shape}")
            self.Y_train = self.Y_train.map({'M': 1, 'B': 0}).astype(int)
            self.Y_test = self.Y_test.map({'M': 1, 'B': 0}).astype(int)
            print(self.Y_train.head(10))
            print(self.Y_test.head(10))
            logger.info(f"Converted Column values M - 1 and B - 0 successfully")

            # HYPOTHESIS TEST
            # The purpose of hypothesis testing is to test whether
            # the null hypothesis (there is no difference, no effect) can be rejected or approved.
            self.training_ind_data, self.testing_ind_data = hypothesis(self.training_ind_data, self.Y_train, self.testing_ind_data, self.Y_test)

            # After removing unwanted columns using h_t to know shape of columns
            logger.info(f"After Removing unwanted columns using Hypothesis testing from train data: {self.training_ind_data.shape}")
            logger.info(f"After removing unwanted columns using Hypothesis testing from test data: {self.testing_ind_data.shape}")
            return self.training_ind_data, self.testing_ind_data, self.Y_train, self.Y_test

        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

    def data_Balancing(self):
        try:
            self.training_ind_data, self.testing_ind_data, self.Y_train, self.Y_test = self.cat_num()
            logger.info("Before Upsampling (SMOTE)")

            logger.info(f"Class 0 Count (Benign): {sum(self.Y_train == 0)}")
            logger.info(f"Class 1 Count (Malignant): {sum(self.Y_train == 1)}")

            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=2)
            self.training_ind_data_up, self.Y_train_up = sm.fit_resample(self.training_ind_data, self.Y_train)

            logger.info("After Upsampling (SMOTE Applied)")
            logger.info(f"Class 0 Count (Benign): {sum(self.Y_train_up == 0)}")
            logger.info(f"Class 1 Count (Malignant): {sum(self.Y_train_up == 1)}")

            return self.training_ind_data_up, self.Y_train_up, self.testing_ind_data, self.Y_test

        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

    # *******---------------------------SCALING_DATA******------------------------------------------------------
    def scaling_data(self):
        try:
            self.training_ind_data_up, self.Y_train_up, self.testing_ind_data, self.Y_test = self.data_Balancing()
            logger.info(f"First 3 rows of training data before scaling:\n{self.training_ind_data_up.head(3)}")
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            sc.fit(self.training_ind_data_up)

            self.scaled_training_inde_cols = sc.transform(self.training_ind_data_up)
            self.scaled_test_inde_cols = sc.transform(self.testing_ind_data)

            logger.info("Feature scaling done using StandardScaler.")
            logger.info(f"Scaled Training Data Sample:\n{self.scaled_training_inde_cols[:3]}")
            #multi_models(self.scaled_training_inde_cols,self.Y_train_up,self.scaled_test_inde_cols,self.Y_test)
            #selecting_best_model(self.scaled_training_inde_cols, self.Y_train_up, self.scaled_test_inde_cols, self.Y_test)
            f_m(self.scaled_training_inde_cols, self.Y_train_up, self.scaled_test_inde_cols, self.Y_test)
            outcomes = testing_()
            logger.info(f"Model prediction  : {outcomes}")
            logger.info(f"columns names : {self.training_ind_data_up.columns}")


        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")


if __name__ == "__main__":
    try:
        reg = BREAST_CANCER("C:\\Users\\geeth\\PycharmProjects\\Breast_Cancer_Prediction\\breast-cancer.csv")
        reg.scaling_data()
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

