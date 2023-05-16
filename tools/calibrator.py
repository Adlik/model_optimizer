# pylint: disable=import-error, unused-import
"""Calibrator.py

The original code could be found in TensorRT-7.x sample code:
"samples/python/int8_caffe_mnist/calibrator.py".  I made the
modification so that the Calibrator could handle MS-COCO dataset
images instead of MNIST.

Create custom calibrator, use to calibrate int8 TensorRT model.
Need to override some methods of trt.IInt8EntropyCalibrator2, such as get_batch_size, get_batch,
read_calibration_cache, write_calibration_cache.
"""
import os
import numpy as np

import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda


# pylint: disable=missing-function-docstring
def load_calibration_data(cali_data_loader):
    """Load  calibration dataloader.

    Args:
        cali_data_loader: dataloader

    Returns:
        dataset (list)
    """
    dataset = []
    for data, _ in cali_data_loader:
        data = data.numpy().astype(np.float32)
        dataset.append(data)
    return dataset


class ImageEntropyCalibrator2(trt.IInt8EntropyCalibrator2):
    """YOLOEntropyCalibrator

    This class implements TensorRT's IInt8EntropyCalibtrator2 interface. Entropy calibration chooses the tensor’s scale
    factor to optimize the quantized tensor’s information-theoretic content, and usually suppresses outliers in the
    distribution. This is the current and recommended entropy calibrator and is required for DLA. Calibration
    happens before Layer fusion by default. It is recommended for CNN-based networks.
    """

    def __init__(self, cali_data_loader, cache_file):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = load_calibration_data(cali_data_loader)
        self.batch_size = self.data[0].shape[0]
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self):
        """
        TensorRT passes along the names of the engine bindings to the get_batch function.
        You don't necessarily have to use them, but they can be useful to understand the order of
        the inputs. The bindings list is expected to have the same ordering as 'names'.
        """
        if self.current_index == len(self.data):
            return None

        batch = self.data[self.current_index].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += 1
        print(f'Calibrate batch = {self.current_index} / {len(self.data)}')
        return [self.device_input]

    def read_calibration_cache(self):
        """
        If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class ImageMixMaxCalibrator(trt.IInt8MinMaxCalibrator):
    """ImageMixMaxCalibrator

    This class implements TensorRT's IInt8MinMaxCalibrator interface.
    This calibrator uses the entire range of the activation distribution to determine the scale factor.
    It seems to work better for NLP tasks. Calibration happens before Layer fusion by default.
    This is recommended for networks such as NVIDIA BERT
    """

    def __init__(self, cali_data_loader, cache_file):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.cache_file = cache_file
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = load_calibration_data(cali_data_loader)
        self.batch_size = self.data[0].shape[0]
        self.current_index = 0
        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self):
        """
        TensorRT passes along the names of the engine bindings to the get_batch function.
        You don't necessarily have to use them, but they can be useful to understand the order of
        the inputs. The bindings list is expected to have the same ordering as 'names'.
        """
        if self.current_index == len(self.data):
            return None

        batch = self.data[self.current_index].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += 1
        print(f'Calibrate batch = {self.current_index} / {len(self.data)}')
        return [self.device_input]

    def read_calibration_cache(self):
        """If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class ImageLegacyCalibrator(trt.IInt8LegacyCalibrator):
    """ImageLegacyCalibrator

    This class implements TensorRT's IInt8LegacyCalibrator interface.
    This calibrator is for compatibility with TensorRT 2.0 EA. This calibrator requires
    user parameterization and is provided as a fallback option if the other calibrators
    yield poor results. Calibration happens after Layer fusion by default. You can customize
    this calibrator to implement percentile max, for example, 99.99% percentile max is
    observed to have best accuracy for NVIDIA BERT.
    """

    def __init__(self, cali_data_loader, cache_file):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8LegacyCalibrator.__init__(self)
        self.cache_file = cache_file
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = load_calibration_data(cali_data_loader)
        self.batch_size = self.data[0].shape[0]
        self.current_index = 0
        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self):
        """
        TensorRT passes along the names of the engine bindings to the get_batch function.
        You don't necessarily have to use them, but they can be useful to understand the order of
        the inputs. The bindings list is expected to have the same ordering as 'names'.
        """
        if self.current_index == len(self.data):
            return None

        batch = self.data[self.current_index].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += 1
        print(f'Calibrate batch = {self.current_index} / {len(self.data)}')
        return [self.device_input]

    def read_calibration_cache(self):
        """If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def get_quantile(self):
        return 0.9999

    def get_regression_cutoff(self):
        return 1.0

    def read_histogram_cache(self):
        return None

    def write_histogram_cache(self):
        return None
