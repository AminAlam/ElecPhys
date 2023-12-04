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
        npz_files_folder = os.path.join(os.path.dirname(__file__), 'data', 'npz')
        output_npz_folder = os.path.join(os.path.dirname(__file__), 'data', 'npz_zscore')
        if os.path.exists(output_npz_folder):
            shutil.rmtree(output_npz_folder)
        preprocessing.zscore_normalize_npz(npz_files_folder, output_npz_folder)
        self.assertTrue(os.path.exists(output_npz_folder))

    def test_normalize_npz(self):
        npz_files_folder = os.path.join(os.path.dirname(__file__), 'data', 'npz')
        output_npz_folder = os.path.join(os.path.dirname(__file__), 'data', 'npz_normalized')
        if os.path.exists(output_npz_folder):
            shutil.rmtree(output_npz_folder)
        preprocessing.normalize_npz(npz_files_folder, output_npz_folder)
        self.assertTrue(os.path.exists(output_npz_folder))


class TestCases_fourier_analysis(unittest.TestCase):
    def test_stft_numeric_output_from_npz(self):
        npz_files_folder = os.path.join(os.path.dirname(__file__), 'data', 'npz')
        output_npz_folder = os.path.join(os.path.dirname(__file__), 'data', 'npz_stft')
        if os.path.exists(output_npz_folder):
            shutil.rmtree(output_npz_folder)
        window_size = 1
        overlap = 0.5
        for window_type in ['hann', 'kaiser 5']:
            fourier_analysis.stft_numeric_output_from_npz(npz_files_folder, output_npz_folder, window_size, overlap, window_type)


class TestCases_visualization(unittest.TestCase):
    def test_plot_stft(self):
        npz_files_folder = os.path.join(os.path.dirname(__file__), 'data', 'npz_stft')
        npz_files = os.listdir(npz_files_folder)
        npz_file = npz_files[0]
        output_plot_file = os.path.join(os.path.dirname(__file__), 'data', 'plots', 'stft_plot.png')
        
        f_min = None
        f_max = None
        t_min = None
        t_max = None
        db_min = None
        db_max = None

        for f_min in [None, 50]:
            for f_max in [None, 200]:
                for t_min in [None, 2]:
                    for t_max in [None, 10]:
                        for db_min in [None, -50]:
                            for db_max in [None, 50]:
                                if os.path.exists(output_plot_file):
                                    os.remove(output_plot_file)
                                visualization.plot_stft_from_npz(os.path.join(npz_files_folder, npz_file), output_plot_file, f_min, f_max, t_min, t_max, db_min, db_max)
                                self.assertTrue(os.path.exists(output_plot_file))
    
    def test_plot_signal(self):
        npz_folder_path = os.path.join(os.path.dirname(__file__), 'data', 'npz')
        output_plot_file = os.path.join(os.path.dirname(__file__), 'data', 'plots', 'signal_plot.png')
        t_min = None
        t_max = None
        channels_list = None
        if os.path.exists(output_plot_file):
            os.remove(output_plot_file)
        visualization.plot_signals_from_npz(npz_folder_path, output_plot_file, t_min, t_max, channels_list)


class TestCases_utils(unittest.TestCase):
    def test_get_matlab_engine(self):
        pass

if __name__ == '__main__':
    unittest.main()