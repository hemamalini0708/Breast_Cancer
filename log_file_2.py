import logging


def Set_levels(acc):
    log = logging.getLogger(acc)
    log.setLevel("DEBUG")

    # Create a script for the data
    handler = logging.FileHandler(f"C:\\Users\\geeth\\PycharmProjects\\Breast_Cancer_Prediction\\log_files_2\\{acc}.log", mode='w')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)

    return log
