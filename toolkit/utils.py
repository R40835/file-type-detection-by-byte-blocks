import os
import math

import numpy as np
import pandas as pd


def get_file_types(directory: str) -> list:
    """
    Get all the files in a directory and their corresponding type.

    Args:
        directory: The directory of interest.
    Returns:
        list: A list of dictionaries with files and types.
    """
    files_data = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            _, extension = os.path.splitext(file_path)
            files_data.append({'file': file, 'type': extension[1::]})
    return files_data


def get_bytes(path: str) -> bytes:
    """
    Gets all the bytes to construct a file.

    Args:
        path (str): The path of the target file.
    Returns:
        bytes: The file bytes.
    """
    with open(path, 'rb') as file:
        binary_data = file.read()
    return binary_data


def get_1st_block_bytes(path: str) -> bytes:
    """
    Gets the 4096 bytes of the first block where a file data are stored.

    Args:
        path (str): The path of targeted file.
    Returns:
        bytes: A sequence of 4096 byte objects
    """
    BLOCK_SIZE_BYTES = 4096
    with open(path, "rb") as file:
        binary_data = file.read(BLOCK_SIZE_BYTES)
    return binary_data


def get_2nd_block_bytes(path: str) -> bytes:
    """
    Gets 4096 bytes of a file body block.

    Args:
        path (str): The path of targeted file.
    Returns:
        bytes: A sequence of 4096 byte objects
    """
    BLOCK_SIZE_BYTES = 4096
    BLOCK_START_IDX  = 4096
    with open(path, "rb") as file:
        file.seek(BLOCK_START_IDX)
        binary_data = file.read(BLOCK_SIZE_BYTES)
    return binary_data


def get_last_block_bytes(path: str, file_size_kb: float) -> bytes:
    """
    Gets the 4096 bytes of the last block where a file data are stored.

    Args:
        path (str): The path of targeted file.
        file_size_kb (int): The file size in KB.
    Returns:
        bytes: A sequence of 4096 byte objects
    """
    BLOCK_SIZE_BYTES = 4096
    file_size_bytes = file_size_kb * 1024
    number_of_blocks = math.ceil(file_size_bytes / BLOCK_SIZE_BYTES)
    BLOCK_START_IDX = (number_of_blocks - 1) * BLOCK_SIZE_BYTES
    with open(path, "rb") as file:
        file.seek(BLOCK_START_IDX)
        binary_data = file.read(4096)
    return binary_data


def print_50bytes_1st_block(df: pd.DataFrame, file_type: str) -> None:
    """
    Prints the first 50 bytes that include file headers for 5 samples.

    Args:
        df (pd.DataFrame): A dataframe containing 'type' and '1st_block_bytes' columns.
        file_type (str): The type of a file. 
    """
    print("First block:")
    for i, bytes_seq in enumerate(df[df["type"] == file_type]["1st_block_bytes"]):
        print(f"sample {i + 1}: {bytes_seq[:50]}")
        if i == 4:
            break
    print("\n")


def print_50bytes_body_block(df: pd.DataFrame, file_type: str) -> None:
    """
    Prints 50 bytes of file bodies for 5 samples.

    Args:
        df (pd.DataFrame): A dataframe containing 'type' and 'body_block_bytes' columns.
        file_type (str): The type of a file.
    """
    print("Body block:")
    for i, bytes_seq in enumerate(df[df["type"] == file_type]["body_block_bytes"]):
        print(f"sample {i + 1}: {bytes_seq[:50]}")
        if i == 4:
            break 
    print("\n")


def print_50bytes_last_block(df: pd.DataFrame, file_type: str) -> None:
    """
    Prints the last 50 bytes that might include file trailers for 5 samples.

    Args:
        df (pd.DataFrame): A dataframe containing 'type' and 'last_block_bytes' columns.
        file_type (str): The type of a file.
    """
    print("Last block:")
    for i, bytes_seq in enumerate(df[df["type"] == file_type]["last_block_bytes"]):
        if set(bytes_seq) != {0}:            
            meaningful_indices = [i for i, byte in enumerate(bytes_seq) if byte != 0]
            start_idx = meaningful_indices[0]
            end_idx = meaningful_indices[-1] + 1
            last_slice = bytes_seq[start_idx:end_idx]            
            print(f"sample {i + 1}: {last_slice[-51:-1]}")
        else:
            print("The sequence contains only null values.")

        if i == 4:
            break 
    print("\n")


def pad_array(arr: np.ndarray, length: int=4096, pad_value: int=260) -> np.ndarray:
    """
    Pads Arrays to make them homogeneous. The padding value passed 
    should not be in the interval 0-255 as we are working with 
    bytes represented as integer values.

    Args:
        arr (np.ndarray): The array to be padded.
        length (int): The length of desire.
        pad_value (int): The padding value.
    Returns:
        np.ndarray: The padded array.
    """
    if len(arr) < length:
        padded_arr = np.concatenate([arr, np.full((length - len(arr),), pad_value)])
    else:
        padded_arr = arr
    return padded_arr


def byte_frequency_histogram(byte_integers: np.ndarray) -> np.ndarray:
    """
    Calculates the frequency of each byte value from 0 to 255.

    Args:
        byte_integers (np.ndarray): An array of integer representations of bytes.
    Returns:
        np.ndarray: An array containing the byte frequencies
    """
    return np.bincount(byte_integers, minlength=256)


def convert_cat2num(file_type: str) -> int:
    """ 
    Converts categorical classes to numerical.

    Args:
        file_type (str): The file type.
    Returns:
        int: The numerical class.
    """
    if file_type == "doc":
        return 1
    elif file_type == "pdf":
        return 2
    elif file_type == "ps":
        return 3
    elif file_type == "xls":
        return 4
    elif file_type == "ppt":
        return 5
    elif file_type == "swf":
        return 6
    elif file_type == "gif":
        return 7
    elif file_type == "jpg":
        return 8
    elif file_type == "png":
        return 9
    elif file_type == "html":
        return 10
    elif file_type == "txt":
        return 11
    elif file_type == "xml":
        return 12