from pathlib import Path
from model import UNet, OptAttNet

"""
config file
"""

# number of classes for each type of problem
num_classes = {
    'binary': 2,
    'parts': 4,
    'instruments': 8
}


# train config
model = OptAttNet
fold = 0 # fold for train/validation split
problem_type = 'binary'
jaccard_weight = 0.3
epoches = 10
batch_size = 2
lr = 1e-5

# directory
root_dir = Path('.')
data_dir = root_dir / 'data'
train_dir = data_dir / 'train'
cropped_train_dir = data_dir / 'cropped_train'
model_dir = root_dir / model.__name__


# data preprocessing
original_height, original_width = 1080, 1920
cropped_height, cropped_width = 1024, 1280
# train_height, train_width = 1024, 1280
train_height, train_width = 64, 80
h_start, w_start = 28, 320


# segmentation
binary_factor = 255
parts_factor = 85
instrument_factor = 32


# GPU and cuda
device_ids = [0,]
num_workers = 0

