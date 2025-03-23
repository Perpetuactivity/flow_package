from unittest import TestCase
import pandas as pd
from flow_package import data_preprocessing
import os

class TestPreprocessing(TestCase):

    def __init__(self):
        super().__init__()
        self.train_path = os.path.abspath("./data/test.csv")
        self.test_path = os.path.abspath("./data/test.csv")

    def test_preprocess_data(self):
        buf_train, buf_test, buf_labels = data_preprocessing(
            self.train_path,
            self.test_path,
            categorical_index=["Protocol", "Destination Port"],
            binary_normal_label="BENIGN",
        )