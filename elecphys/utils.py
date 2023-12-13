import subprocess
import os
import sys
import re

def get_matlab_engine():
    """Installs MATLAB engine for Python
    input:
    output:
        eng: MATLAB engine for Python
    """
    matlab_installation_url = 'https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html'
    try:
        import matlab.engine
    except:
        raise ImportError(f'MATLAB engine for Python not installed. Please install MATLAB engine for Python from {matlab_installation_url}')        
    
    return matlab.engine.start_matlab()


def sort_file_names(file_names):
    """Sorts file names in ascending order
    input:
        file_names: list of file names - type: list
    output:
        file_names: sorted list of file names - type: list
    """
    file_names.sort(key=lambda f: int(re.sub('\D', '', f)))
    return file_names


def convert_string_to_list(string):
    """Converts string to list
    input:
        string: string to be converted - type: str
    output:
        list: converted list - type: list
    """
    if type(string) is not str:
        return string
    string = string.replace('[', '')
    string = string.replace(']', '')
    string = string.replace(' ', '')
    string = string.split(',')
    return list(map(int, string))
