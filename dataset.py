import os
import numpy as np
import torch
import tifffile

from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image

from utils import *


class TwoSliceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.preloaded_data = {}  # To store preloaded data
        self.pairs = self.preload_and_make_pairs(root_folder_path)

    def preload_and_make_pairs(self, root_folder_path):
        pairs = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume  # Preload data here
                num_slices = volume.shape[0]
                for i in range(num_slices - 1):  # Ensure there is a next slice
                    input_slice_index = i
                    target_slice_index = i + 1
                    pairs.append((full_path, input_slice_index, target_slice_index))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        file_path, input_slice_index, target_slice_index = self.pairs[index]
        
        # Access preloaded data instead of reading from file
        input_slice = self.preloaded_data[file_path][input_slice_index][..., np.newaxis]
        target_slice = self.preloaded_data[file_path][target_slice_index][..., np.newaxis]

        if self.transform:
            input_slice, target_slice = self.transform((input_slice, target_slice))

        return input_slice, target_slice


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.preloaded_data = {}  # To store preloaded data
        self.pairs = self.preload_and_make_pairs(root_folder_path)

    def preload_and_make_pairs(self, root_folder_path):
        pairs = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume  # Preload data here
                num_slices = volume.shape[0]
                for i in range(num_slices - 1):  # Ensure there is a next slice
                    input_slice_index = i
                    target_slice_index = i + 1
                    pairs.append((full_path, input_slice_index, target_slice_index))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        file_path, input_slice_index, target_slice_index = self.pairs[index]
        
        # Access preloaded data instead of reading from file
        input_slice = self.preloaded_data[file_path][input_slice_index][..., np.newaxis]
        target_slice = self.preloaded_data[file_path][target_slice_index][..., np.newaxis]

        plot_intensity_line_distribution(input_slice)

        if self.transform:
            input_slice = self.transform((input_slice, target_slice))

        return input_slice