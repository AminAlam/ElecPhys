import os
import sys
import unittest
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'lib'))
import conversion
import preprocessing
import fourier_analysis
import visualization
import data_loading

class TestCases_conversion(unittest.TestCase):
    def test_1_rhd_to_mat(self):   
        folder_path = os.path.join(os.path.dirname(__file__), 'data', 'rhd')
        output_mat_file = os.path.join(os.path.dirname(__file__), 'data', 'mat', 'sample.mat')
        for ds_factor in [1, 20]:
            if os.path.exists(output_mat_file):
                os.remove(output_mat_file)
            conversion.convert_rhd_to_mat_matlab(folder_path, output_mat_file, ds_factor)
            self.assertTrue(os.path.exists(output_mat_file))

    def test_2_mat_to_npz(self):
        mat_file = os.path.join(os.path.dirname(__file__), 'data', 'mat', 'sample.mat')
        output_npz_folder = os.path.join(os.path.dirname(__file__), 'data', 'npz')
        for notch_filter_freq in [0, 50, 60]:
            if os.path.exists(output_npz_folder):
                shutil.rmtree(output_npz_folder)
            conversion.convert_mat_to_npz_matlab(mat_file, output_npz_folder, notch_filter_freq)
            self.assertTrue(os.path.exists(output_npz_folder))


    

class TestCases_preprocessing(unittest.TestCase):
    def test_apply_notch(self):
        npz_files_folder = os.path.join(os.path.dirname(__file__), 'data', 'npz')
        npz_files = os.listdir(npz_files_folder)
        npz_file = npz_files[0]
        _signal_chan, fs = data_loading.load_npz(os.path.join(npz_files_folder, npz_file))
        output = preprocessing.apply_notch(_signal_chan, {'Q':60, 'fs':fs, 'f0':50})
        self.assertTrue(output.shape == _signal_chan.shape)

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


class TestCases_utils(unittest.TestCase):
    def test_get_matlab_engine(self):
        pass

if __name__ == '__main__':
    unittest.main()