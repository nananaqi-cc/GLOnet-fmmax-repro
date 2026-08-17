"""General utility functions"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import json
import logging
import csv
import scipy.io as io
import torch
import numpy as np

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def row_csv2dict(csv_file):
    dict_club={}
    with open(csv_file)as f:
        reader=csv.reader(f,delimiter=',')
        for row in reader:
            dict_club[(row[0],row[1])]=row[2]
    return dict_club


def save_checkpoint(state, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'model.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    # Write in the destination directory and atomically replace the final file.
    # A power loss or interrupted process then leaves either the previous valid
    # checkpoint or a disposable temporary file, rather than a truncated model.
    temp_filepath = filepath + f'.tmp.{os.getpid()}'
    torch.save(state, temp_filepath)
    os.replace(temp_filepath, filepath)



def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)

    model.load_state_dict(checkpoint['gen_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def plot_loss_history(loss_history, params, per_wl_history=None, wavelengths=None):
    # backward-compat: accept either 3-tuple (legacy) or 4-tuple (with sign-eff)
    if len(loss_history) == 4:
        effs_mean_history, diversity_history, binarization_history, effs_sign_history = loss_history
    else:
        effs_mean_history, diversity_history, binarization_history = loss_history
        effs_sign_history = None

    iterations = [i * params.plot_iter for i in range(len(effs_mean_history))]

    # --- Main training curves ---
    plt.figure()
    plt.plot(iterations, effs_mean_history, label='eff_mean (raw)')
    if effs_sign_history is not None:
        plt.plot(iterations, effs_sign_history, label='eff_mean (sign)')
    plt.plot(iterations, diversity_history, label='Pattern diversity')
    plt.plot(iterations, binarization_history, label='Binarization')
    plt.xlabel('iteration')
    plt.legend()
    plt.axis([0, len(effs_mean_history) * params.plot_iter, 0, 1.05])
    plt.savefig(os.path.join(params.output_dir, 'figures', 'Train_history.png'))
    plt.close()

    # --- Per-wavelength curves (multi-wavelength only) ---
    if per_wl_history is not None and wavelengths is not None:
        per_wl_arr = np.asarray(per_wl_history)  # (n_evals, n_wavelengths)
        plt.figure()
        for j, wl in enumerate(wavelengths):
            plt.plot(iterations, per_wl_arr[:, j], label=f'{wl:.0f} nm')
        plt.xlabel('iteration')
        plt.ylabel('eff_mean (sign)')
        plt.legend()
        plt.axis([0, len(effs_mean_history) * params.plot_iter, 0, 1.05])
        plt.savefig(os.path.join(params.output_dir, 'figures', 'Train_history_per_wl.png'))
        plt.close()

    # --- Save to .mat ---
    history_path = os.path.join(params.output_dir, 'history.mat')
    mdict = {'effs_mean_history':    np.asarray(effs_mean_history),
             'diversity_history':    np.asarray(diversity_history),
             'binarization_history': np.asarray(binarization_history)}
    if effs_sign_history is not None:
        mdict['effs_sign_history'] = np.asarray(effs_sign_history)
    if per_wl_history is not None:
        mdict['per_wl_history'] = np.asarray(per_wl_history)
    temp_history_path = history_path + f'.tmp.{os.getpid()}'
    io.savemat(temp_history_path, mdict=mdict, appendmat=False)
    os.replace(temp_history_path, history_path)
         

def plot_histogram(Effs, Iter, fig_path, title_suffix=None):
    ax = plt.figure()
    bins = [i*5 for i in range(21)]
    plt.hist(Effs*100, bins, facecolor='blue', alpha=0.5)
    plt.xlim(0, 100)
    plt.ylim(0, 50)
    plt.yticks([])
    plt.xticks(fontsize=12)
    plt.xlabel('Deflection efficiency (%)', fontsize=12)
    title = f'Iteration {Iter}'
    if title_suffix:
        title = f'{title}  ({title_suffix})'
    plt.title(title, fontsize=16)
    plt.savefig(fig_path, dpi=300)
    plt.close()


