import re
import numpy as np


def get_matlab_engine():
    """Installs MATLAB engine for Python

        Parameters
        ----------

        Returns
        ----------
        eng: matlab.engine
            MATLAB engine for Python
    """
    matlab_installation_url = 'https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html'
    try:
        import matlab.engine
    except BaseException:
        raise ImportError(
            f'MATLAB engine for Python not installed. Please install MATLAB engine for Python from {matlab_installation_url}')

    return matlab.engine.start_matlab()


def sort_file_names(file_names: list) -> list:
    """Sorts file names in ascending order

        Parameters
        ----------
        file_names: list
            list of file names to be sorted

        Returns
        ----------
        file_names: list
            sorted list of file names
    """
    file_names = keep_npz_files(file_names)
    file_names_with_number = []
    file_names_without_number = []
    for file_name in file_names:
        if re.search('\\d', file_name):
            file_names_with_number.append(file_name)
        else:
            file_names_without_number.append(file_name)
    file_names_with_number.sort(key=lambda f: int(re.sub('\\D', '', f)))
    file_names_without_number.sort()
    file_names = file_names_with_number + file_names_without_number
    return file_names


def keep_npz_files(file_names: list) -> list:
    """Keeps only NPZ files

        Parameters
        ----------
        file_names: list
            list of file names

        Returns
        ----------
        file_names: list
            list of NPZ file names
    """
    file_names = [
        file_name for file_name in file_names if file_name.endswith('.npz')]
    return file_names


def remove_non_numeric(input_list: list) -> list:
    """Removes None values from list

        Parameters
        ----------
        input_list: list
            list to be processed

        Returns
        ----------
        output_list: list
            processed list
    """

    # regex to remove non-numeric characters
    output_list = []
    for item in input_list:
        if isinstance(item, str):
            item = re.sub('[^0-9]', '', item)
            if item != '':
                output_list.append(int(item))
        else:
            output_list.append(item)
    return output_list


def convert_string_to_list(string):
    """Converts string to list

        Parameters
        ----------
        string: str
            string to be converted

        Returns
        ----------
        output: list
            converted list
    """
    if string is None or string == '' or string == 'None':
        return None
    if not isinstance(string, str):
        string = remove_non_numeric(string)
        return string
    string = string.replace('[', '')
    string = string.replace(']', '')
    string = string.replace(' ', '')
    string = string.split(',')
    output = list(map(int, string))
    output = np.unique(output)
    output = output.tolist()
    return output


def check_freq_bands(freq_bands: list, fs: int) -> list:
    """ checks frequency bands in the tuple to see if they are valid such as first element is smaller than second element and both of them are smaller than half of sampling frequency

        Parameters
        ----------
        freq_bands: tuple
            frequency bands to be checked

        Returns
        ----------
        freq_bands: tuple
            checked frequency bands
    """
    for freq_band in freq_bands:
        if freq_band[0] > freq_band[1]:
            raise ValueError(
                f'Invalid frequency band: {freq_band}. First element should be smaller than second element.')
        if freq_band[0] > fs / 2 or freq_band[1] > fs / 2:
            raise ValueError(
                f'Invalid frequency band: {freq_band}. Both elements should be smaller than half of sampling frequency (fs={fs}Hz).')

    return freq_bands
