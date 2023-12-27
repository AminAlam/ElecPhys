import utils
import os


def pca_from_npz(input_npz_folder, output_npz_folder,
                 n_components, matrix_whitenning, channels_list) -> None:
    """ Performs PCA on NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_npz_folder: str
            path to output npz folder
        n_components: int
            number of components to keep
        matrix_whitenning: bool
            whether to whiten the matrix
        channels_list: list
            list of channels to plot

        Returns
        ----------
    """
    npz_files = os.listdir(input_npz_folder)
    npz_files = utils.sort_file_names(npz_files)

    if channels_list is None:
        channels_list = tuple(range(1, len(npz_files) + 1))
    else:
        channels_list = sorted(channels_list)

    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')

    print(f'--- Performing PCA on NPZ files...')
