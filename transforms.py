import numpy as np
import os
import sys
from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image
import torch
from sklearn.preprocessing import PowerTransformer

class Normalize(object):
    """
    Normalize an image using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (tuple): Containing input and target images to be normalized.
        
        Returns:
            Tuple: Normalized input and target images.
        """
        input_img, target_img = data

        # Normalize input image
        input_normalized = (input_img - self.mean) / self.std

        # Normalize target image
        target_normalized = (target_img - self.mean) / self.std

        return input_normalized, target_normalized


class MinMaxNormalize(object):
    """
    Normalize images to the 0-1 range using global minimum and maximum values provided at initialization.
    """

    def __init__(self, global_min, global_max):
        """
        Initializes the normalizer with global minimum and maximum values.

        Parameters:
        - global_min (float): The global minimum value used for normalization.
        - global_max (float): The global maximum value used for normalization.
        """
        self.global_min = global_min
        self.global_max = global_max

    def __call__(self, data):
        """
        Normalize input and target images to the 0-1 range using the global min and max.

        Args:
            data (tuple): Containing input and target images to be normalized.

        Returns:
            Tuple: Normalized input and target images.
        """
        input_img, target_img = data

        # Normalize input image
        input_normalized = (input_img - self.global_min) / (self.global_max - self.global_min)
        input_normalized = np.clip(input_normalized, 0, 1)  # Ensure within [0, 1] range

        # Normalize target image
        target_normalized = (target_img - self.global_min) / (self.global_max - self.global_min)
        target_normalized = np.clip(target_normalized, 0, 1)  # Ensure within [0, 1] range

        return input_normalized.astype(np.float32), target_normalized.astype(np.float32)


class LogScaleAndNormalize(object):
    """
    Apply logarithmic scaling followed by Z-score normalization to a single-channel image.

    Args:
        mean (float): Mean of the log-scaled data.
        std (float): Standard deviation of the log-scaled data.
        epsilon (float): A small value added to the input to avoid logarithm of zero.

    """

    def __init__(self, mean, std, epsilon=1e-10):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def __call__(self, data):
        """
        Apply logarithmic scaling followed by Z-score normalization to a single-channel image with dimensions (1, H, W).

        Args:
            img (numpy.ndarray): Image to be transformed, expected to be in the format (1, H, W).

        Returns:
            numpy.ndarray: Transformed image.
        """

        input_img, target_img = data

        log_scaled_mean = np.log(self.mean + self.epsilon)
        log_scaled_std = np.log(self.std + self.epsilon)

        log_scaled_input_img = np.log(input_img + self.epsilon)
        log_scaled_target_img = np.log(target_img + self.epsilon)
        

        normalized_input_img = (log_scaled_input_img - log_scaled_mean) / log_scaled_std
        normalized_target_img = (log_scaled_target_img - log_scaled_mean) / log_scaled_std


        return normalized_input_img, normalized_target_img


class PowerTransform(object):
    """
    Apply the Yeo-Johnson power transformation to normalize images.
    
    Args:
        method (str): Method of power transformation ('yeo-johnson' or 'box-cox').
    """

    def __init__(self, method='yeo-johnson'):
        self.transformer = PowerTransformer(method=method, standardize=True)

    def __call__(self, data):
        """
        Apply Yeo-Johnson transformation to both input and target images in the tuple.
        
        Args:
            data (tuple): Containing input and target images to be transformed.
        
        Returns:
            Tuple: Transformed input and target images.
        """
        input_img, target_img = data
        
        # Check if images need reshaping for transformation
        original_shape_input = input_img.shape
        original_shape_target = target_img.shape
        
        # Reshape if necessary (PowerTransformer expects 2D arrays)
        if input_img.ndim == 3:
            input_img = input_img.reshape(-1, original_shape_input[1] * original_shape_input[2]).T
        if target_img.ndim == 3:
            target_img = target_img.reshape(-1, original_shape_target[1] * original_shape_target[2]).T
        
        # Apply transformation
        input_transformed = self.transformer.fit_transform(input_img)
        target_transformed = self.transformer.fit_transform(target_img)
        
        # Reshape back to original shape
        input_transformed = input_transformed.T.reshape(original_shape_input)
        target_transformed = target_transformed.T.reshape(original_shape_target)

        return input_transformed, target_transformed




class RandomFlip(object):

    def __call__(self, data):

        input_img, target_img = data

        if np.random.rand() > 0.5:
            input_img = np.fliplr(input_img)
            target_img = np.fliplr(target_img)

        if np.random.rand() > 0.5:
            input_img = np.flipud(input_img)
            target_img = np.flipud(target_img)

        return input_img, target_img
    

class RandomHorizontalFlip:
    def __call__(self, data):
        """
        Apply random horizontal flipping to both the input stack of slices and the target slice.
        In 50% of the cases, only horizontal flipping is applied without vertical flipping.
        
        Args:
            data (tuple): A tuple containing the input stack and the target slice.
        
        Returns:
            Tuple: Horizontally flipped input stack and target slice, if applied.
        """
        input_stack, target_slice = data

        # Apply horizontal flipping with a 50% chance
        if np.random.rand() > 0.5:
            # Flip along the width axis (axis 1), keeping the channel dimension (axis 2) intact
            input_stack = np.flip(input_stack, axis=1)
            target_slice = np.flip(target_slice, axis=1)

        # With the modified requirements, we remove the vertical flipping part
        # to ensure that only horizontal flipping is considered.

        return input_stack, target_slice




class RandomCrop:
    def __init__(self, output_size=(64, 64)):
        """
        RandomCrop constructor for cropping both the input stack of slices and the target slice.
        Args:
            output_size (tuple): The desired output size (height, width).
        """
        self.output_size = output_size

    def __call__(self, data):
        """
        Apply the cropping operation.
        Args:
            data (tuple): A tuple containing the input stack and the target slice.
        Returns:
            Tuple: Cropped input stack and target slice.
        """
        input_stack, target_slice = data

        h, w, _ = input_stack.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        input_cropped = input_stack[top:top+new_h, left:left+new_w, :]
        target_cropped = target_slice[top:top+new_h, left:left+new_w, :]

        return (input_cropped, target_cropped)
    



class AddGaussianNoise:
    """
    Add Gaussian noise to both input and target slices.

    Args:
        std_dev_range (tuple): Range for standard deviation (min, max) of the Gaussian noise.
        noise_shift_range (tuple): Range for mean shift (min, max) of the Gaussian noise.
    """

    def __init__(self, std_dev_range=(0.01, 0.05), noise_shift_range=(-0.1, 0.1)):
        self.std_dev_range = std_dev_range
        self.noise_shift_range = noise_shift_range

    def add_gaussian_noise(self, image, std_dev, mean_shift):
        """Helper method to add Gaussian noise to an image."""
        noise = np.random.normal(mean_shift, std_dev, image.shape)
        return image + noise

    def __call__(self, data):
        """
        Apply Gaussian noise to both input and target slices.

        Args:
            data (tuple): Tuple containing (input_slices, target_slice).
                - input_slices (numpy.ndarray): Input slices with dimensions (H, W, 1).
                - target_slice (numpy.ndarray): Target slice with dimensions (H, W, 1).

        Returns:
            tuple: Noisy (input_slices, target_slice).
        """
        input_slice, target_slice = data

        # Determine the maximum value in both input and target slices for consistent noise scaling
        max_value = max(np.max(input_slice), np.max(target_slice))

        # Sample standard deviation and mean shift for noise
        std_dev = np.random.uniform(*self.std_dev_range) * max_value
        mean_shift = np.random.uniform(*self.noise_shift_range) * max_value

        # Apply noise independently to input slices and the target slice
        noisy_input_slice = self.add_gaussian_noise(input_slice, std_dev, mean_shift)
        noisy_target_slice = self.add_gaussian_noise(target_slice, std_dev, mean_shift)

        return noisy_input_slice, noisy_target_slice


    


class CropToMultipleOf32Inference(object):
    """
    Crop each slice in a stack of images to ensure their height and width are multiples of 32.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, data):
        """
        Args:
            stack (numpy.ndarray): Stack of images to be cropped, with shape (H, W, Num_Slices).

        Returns:
            numpy.ndarray: Stack of cropped images.
        """

        stack = data[0]
        h, w, num_slices = stack.shape  # Assuming stack is a numpy array with shape (H, W, Num_Slices)

        # Compute new dimensions to be multiples of 32
        new_h = h - (h % 32)
        new_w = w - (w % 32)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Generate indices for cropping
        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        # Crop each slice in the stack
        cropped_stack = np.zeros((new_h, new_w, num_slices), dtype=stack.dtype)
        for i in range(num_slices):
            cropped_stack[:, :, i] = stack[id_y, id_x, i].squeeze()

        return cropped_stack
    

class CropToMultipleOf16Inference(object):
    """
    Crop each slice in a stack of images to ensure their height and width are multiples of 32.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, data):
        """
        Args:
            stack (numpy.ndarray): Stack of images to be cropped, with shape (H, W, Num_Slices).

        Returns:
            numpy.ndarray: Stack of cropped images.
        """

        stack = data[0]
        h, w, num_slices = stack.shape  # Assuming stack is a numpy array with shape (H, W, Num_Slices)

        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Generate indices for cropping
        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        # Crop each slice in the stack
        cropped_stack = np.zeros((new_h, new_w, num_slices), dtype=stack.dtype)
        for i in range(num_slices):
            cropped_stack[:, :, i] = stack[id_y, id_x, i].squeeze()

        return cropped_stack




class ToTensor(object):
    """
    Convert images or batches of images to PyTorch tensors, handling both single images
    and tuples of images (input_img, target_img). The input is expected to be in the format
    (b, h, w, c) for batches or (h, w, c) for single images, and it converts them to
    PyTorch's (b, c, h, w) format or (c, h, w) for single images.
    """

    def __call__(self, data):
        """
        Convert input images or a tuple of images to PyTorch tensors, adjusting the channel position.

        Args:
            data (numpy.ndarray or tuple of numpy.ndarray): The input can be a single image (h, w, c),
            a batch of images (b, h, w, c), or a tuple of (input_img, target_img) in similar formats.

        Returns:
            torch.Tensor or tuple of torch.Tensor: The converted image(s) as PyTorch tensor(s) in the
            format (c, h, w) for single images or (b, c, h, w) for batches. If input is a tuple, returns
            a tuple of tensors.
        """
        def convert_image(img):
            # Convert a single image or a batch of images to a tensor, adjusting channel position
            if img.ndim == 4:  # Batch of images (b, h, w, c)
                return torch.from_numpy(img.transpose(0, 3, 1, 2).astype(np.float32))
            elif img.ndim == 3:  # Single image (h, w, c)
                return torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
            else:
                raise ValueError("Unsupported image format: must be (h, w, c) or (b, h, w, c).")

        # Check if the input is a tuple of images
        if isinstance(data, tuple):
            return tuple(convert_image(img) for img in data)
        else:
            return convert_image(data)




    
    

    
class BackTo01Range(object):
    """
    Normalize a tensor to the range [0, 1] based on its own min and max values.
    """

    def __call__(self, tensor):
        """
        Args:
            tensor: A tensor with any range of values.
        
        Returns:
            A tensor normalized to the range [0, 1].
        """
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Avoid division by zero in case the tensor is constant
        if (max_val - min_val).item() > 0:
            # Normalize the tensor to [0, 1] based on its dynamic range
            normalized_tensor = (tensor - min_val) / (max_val - min_val)
        else:
            # If the tensor is constant, set it to a default value, e.g., 0, or handle as needed
            normalized_tensor = tensor.clone().fill_(0)  # Here, setting all values to 0

        return normalized_tensor



class DenormalizeAndClip16Bit(object):
    """
    Denormalize an image using mean and standard deviation, and clip values to the 16-bit range.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        # 16-bit range is 0 to 65535
        self.min_value = 0
        self.max_value = 65535

    def __call__(self, img):
        """
        Denormalize an image and clip it to the 16-bit range.
        
        Args:
            img (numpy array): Normalized image to be denormalized.
        
        Returns:
            numpy array: Denormalized and clipped image.
        """
        # Denormalize the image
        img_denormalized = (img * self.std) + self.mean

        # Clip values to the 16-bit range
        img_clipped = np.clip(img_denormalized, a_min=self.min_value, a_max=self.max_value)

        return img_clipped




class ToNumpy(object):

    def __call__(self, data):

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    

    
class Denormalize(object):
    """
    Denormalize an image using mean and standard deviation, then convert it to 16-bit format.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        """
        Initialize with mean and standard deviation.
        
        Args:
            mean (float or tuple): Mean for each channel.
            std (float or tuple): Standard deviation for each channel.
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        """
        Denormalize the image and convert it to 16-bit format.
        
        Args:
            img (numpy array): Normalized image.
        
        Returns:
            numpy array: Denormalized 16-bit image.
        """
        # Denormalize the image by reversing the normalization process
        img_denormalized = (img * self.std) + self.mean

        # Scale the image to the range [0, 65535] and convert to 16-bit unsigned integer
        img_16bit = img_denormalized.astype(np.uint16)
        
        return img_16bit