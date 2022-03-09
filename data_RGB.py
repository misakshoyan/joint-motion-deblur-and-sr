import os
from dataset_RGB import DataLoaderTrain, DataLoaderVal, DataLoaderTrainDeblur, DataLoaderValDeblur, DataLoaderValJoint, DataLoaderTest
#from dataset_hf5 import DataSet, DataValSet

def get_training_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir)

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir)

def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir)

def get_training_data_deblur(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrainDeblur(rgb_dir)

def get_validation_data_deblur(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderValDeblur(rgb_dir)

def get_test_data_deblur(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderValDeblur(rgb_dir)

def get_validation_data_joint(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderValJoint(rgb_dir)

# def get_training_data(h5py_file_path):
#     assert os.path.exists(h5py_file_path)
#     print(os.path.isdir(h5py_file_path), "  ", h5py_file_path)
#     return DataSet(h5py_file_path)
#
# def get_validation_data(root_dir):
#     assert os.path.exists(root_dir)
#     return DataValSet(root_dir)
#
# def get_test_data(rgb_dir, img_options):
#     assert os.path.exists(rgb_dir)
#     return DataLoaderTest(rgb_dir, img_options)
