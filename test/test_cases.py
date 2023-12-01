import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'lib'))
import conversion
import preprocessing
import fourier_analysis
import visualization

class TestCases_conversion(unittest.TestCase):
    def rhd_to_mat(self):   
        pass

    def mat_to_npz(self):
        pass
        

class TestCases_preprocessing(unittest.TestCase):
    def test_apply_notch(self):
        pass

    def test_zscore_normalize_npz(self):
        pass

    def test_normalize_npz(self):
        pass


class TestCases_fourier_analysis(unittest.TestCase):
    def test_stft_numeric_output_from_npz(self):
        pass

    def test_stft_from_array(self):
        pass


class TestCases_visualization(unittest.TestCase):
    def plot_stft():
        pass