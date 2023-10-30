# imports
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import os

# typing hints
from typing import List
from collections.abc import Callable

# used for bpps
from arnie.bpps import bpps

# according to kaggle, this is the maximum # of reactivites to be used
NUM_REACTIVITIES = 457

# there are 4 different bases (AUCG)
NUM_BASES = 4


# encode inputs as
# A : 1
# U : 2
# C : 3
# G : 4
base_map = {
    "A": 1,
    "U": 2,
    "C": 3,
    "G": 4,
}


def filter_data(out: str, key: str, value: str, file_name: str, force: bool):
    """
    Filters a file to only take datapoints
    whose values of `key` are `value`.

    Parameters:
        - out: str - the name of the file that will store the filtered datapoints
        - key: str - the name of the key to look at
        - value: str - the value that the key should have
        - file_name: str - the name of the file that contains all the datapoints.
        - force: bool - whether or not to force re-processing of the data (if False and `out` already exists, no work will be done)
    """
    if os.path.exists(out) and not force:
        print("File already exists, not doing any work")
        return

    count = 0

    # count how many lines we have in total
    with open(file_name) as file:
        line = file.readline()  # ignore the header
        line = (
            file.readline()
        )  # take the first line since we increment count in the loop
        while line != "":
            count += 1
            line = file.readline()

    # use that knowledge for a progress bar
    with open(file_name, "r") as file, open(out, "w") as outfile:
        # write the header
        header = file.readline()
        outfile.write(header)

        # get what index the SN_filter is
        SN_idx = header.split(",").index(key)

        # only take the approved filtered lines
        for _ in tqdm(range(count)):
            line = file.readline()
            temp = line.split(",")
            if temp[SN_idx] == value:
                outfile.write(line)


def filter_train_data(force: bool = False):
    """
    Filters the immense train_data.csv to only take datapoints
    whose SN_filter (Signal to Noise filter) is 1. In other words,
    we only take good reads. These filtered datapoints are then
    written to the file provided

    Parameters:
        - force: bool - whether or not to force re-processing of the data (if False and `out` already exists, no work will be done)
    """
    filter_data("train_data_filtered.csv", "SN_filter", "1", "train_data.csv", force)


def filter_2A3(force: bool = False):
    """
    Only take the 2A3 points

    Parameters:
        - force: bool - whether or not to force re-processing of the data (if False and `out` already exists, no work will be done)
    """
    filter_data(
        "train_data_2a3.csv",
        "experiment_type",
        "2A3_MaP",
        "train_data_filtered.csv",
        force,
    )


def filter_DMS(force: bool = False):
    """
    Only take the DMS points

    Parameters:
        - force: bool - whether or not to force re-processing of the data (if False and `out` already exists, no work will be done)
    """
    filter_data(
        "train_data_dms.csv",
        "experiment_type",
        "DMS_MaP",
        "train_data_filtered.csv",
        force,
    )


def process_data(row):
    """
    Convert a row containing all csv columns in the original dataset
    to a row containing only the columns:
    - inputs
    - outputs
    - bpp
    - output_masks
    - reactivity_error
    - bool_output_masks
    """
    # initialize arrays
    # note that we assume everything is masked until told otherwise
    inputs = np.zeros((NUM_REACTIVITIES,), dtype=np.float32)
    bpp = np.zeros((NUM_REACTIVITIES,), dtype=np.float32)
    output_masks = np.ones((NUM_REACTIVITIES,), dtype=np.bool_)
    reactivity_errors = np.zeros((NUM_REACTIVITIES,), dtype=np.float32)
    reactivities = np.zeros((NUM_REACTIVITIES,), dtype=np.float32)

    seq_len = len(row["sequence"])

    # encode the bases
    inputs[:seq_len] = np.array(
        list(map(lambda letter: base_map[letter], row["sequence"]))
    )

    # get the probability that any of those bases are paired
    bpp[:seq_len] = np.sum(
        bpps(row["sequence"], package="contrafold", linear=True, threshknot=True),
        axis=-1,
    )

    # get the reactivities and their errors
    reactivities[:seq_len] = np.array(
        list(
            map(
                lambda seq_idx: np.float32(
                    row["reactivity_" + str(seq_idx + 1).rjust(4, "0")]
                ),
                range(seq_len),
            )
        )
    )
    reactivity_errors[:seq_len] = np.array(
        list(
            map(
                lambda seq_idx: np.float32(
                    row["reactivity_error_" + str(seq_idx + 1).rjust(4, "0")]
                ),
                range(seq_len),
            )
        )
    )

    # replace reactivity error nans with 0s (assume no error)
    reactivity_errors = np.where(np.isnan(reactivity_errors), 0.0, reactivity_errors)

    # get where all the reactivities are nan
    nan_locats = np.isnan(reactivities)

    # where it is nan, store True, else False
    output_masks[:seq_len] = nan_locats[:seq_len]

    # where it is not nan, store the reactivity and error, else 0
    reactivities[:seq_len] = np.where(
        nan_locats[:seq_len] == False, reactivities[:seq_len], 0.0
    )
    reactivity_errors[:seq_len] = np.where(
        nan_locats[:seq_len] == False, reactivity_errors[:seq_len], 0.0
    )

    # store the values
    row = {}
    row["inputs"] = inputs
    row["bpp"] = bpp
    row["outputs"] = np.clip(reactivities, 0, 1)
    row["output_masks"] = np.clip(
        np.where(output_masks, 0.0, 1.0) - np.abs(reactivity_errors), 0, 1
    )
    row["bool_output_masks"] = output_masks
    row["reactivity_errors"] = np.abs(reactivity_errors)

    return row


def process_data_test(row):
    """
    Almost the same as process_data, except it only takes inputs and bpp
    """
    # initialize arrays
    # note that we assume everything is masked until told otherwise
    inputs = np.zeros((NUM_REACTIVITIES,), dtype=np.float32)
    bpp = np.zeros((NUM_REACTIVITIES,), dtype=np.float32)

    seq_len = len(row["sequence"])

    # encode the bases
    inputs[:seq_len] = np.array(
        list(map(lambda letter: base_map[letter], row["sequence"]))
    )

    # get the probability that any of those bases are paired
    bpp[:seq_len] = np.sum(
        bpps(row["sequence"], package="contrafold", linear=True, threshknot=True),
        axis=-1,
    )

    row["inputs"] = inputs
    row["bpp"] = bpp
    return row


def preprocess_csv(
    out: str,
    file_name: str,
    n_proc: int = 56,
    map_fn: Callable = process_data,
    extra_cols_to_keep: List[str] = [],
    force: bool = False,
):
    """
    Preprocess the csv and save the preprocessed data as a dataset
    that can be loaded via datasets.Dataset.load_from_file

    The dataset contains the following items:
        - bool_output_masks: Tensor(dtype=torch.bool) - the output masks.
            If True, then that item should NOT be used to calculate loss.
            If False, then that item should be used to calculate loss
        - reactivity_errors: Tensor(dtype=torch.float32) - the reactivity errors
        - output_masks: Tensor(dtype=torch.float32) - the elementwise weights to multiply the loss by to properly
            account for masked items and reactivity errors
        - inputs: tensor(dtype=torch.float32) - the input sequence, specifically of shape (None, NUM_REACTIVITIES)
        - bpp: tensor(dtype=torch.float32)
        - outputs: tensor(dtype=torch.float32) - the expected reactivities. Note that a simple MAE or MSE loss will not
            suffice for training models on this dataset. Please use the output_masks tensor as well.

    Parameters:
        - out: str - the name of the file to save the arrays to
        - file_name: str - the name of the input csv file
        - n_proc: int - the number of processes to use while processing data
        - map_fn: Callable - the function to apply to all dataset rows
        - extra_cols_to_keep: List[str] - the names of any extra columns to keep in the dataset
    """
    if os.path.exists(out) and not force:
        print("File already exists, not doing any work.\n")
        return

    names_to_keep = [
        "reactivity_errors",
        "bool_output_masks",
        "output_masks",
        "inputs",
        "outputs",
        "bpp",
    ] + extra_cols_to_keep

    # load dataset and map it to our preprocess function
    ds = Dataset.from_csv(file_name).map(
        map_fn, num_proc=n_proc, load_from_cache_file=not force
    )

    # drop excess columns and save to disk
    ds.remove_columns(
        list(filter(lambda c: c not in names_to_keep, ds.column_names))
    ).save_to_disk(out)
