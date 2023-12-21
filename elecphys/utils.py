import subprocess
import os
import sys
import re

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
    except:
        raise ImportError(f'MATLAB engine for Python not installed. Please install MATLAB engine for Python from {matlab_installation_url}')        
    
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
    file_names_with_number = []
    file_names_without_number = []
    for file_name in file_names:
        if re.search('\d', file_name):
            file_names_with_number.append(file_name)
        else:
            file_names_without_number.append(file_name)
    file_names_with_number.sort(key=lambda f: int(re.sub('\D', '', f)))
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
    file_names = [file_name for file_name in file_names if file_name.endswith('.npz')]
    return file_names


def convert_string_to_list(string):
    """Converts string to list

        Parameters
        ----------
        string: str
            string to be converted
    
        Returns
        ----------
        list: list
            converted list
    """
    if type(string) is not str:
        return string
    string = string.replace('[', '')
    string = string.replace(']', '')
    string = string.replace(' ', '')
    string = string.split(',')
    return list(map(int, string))
