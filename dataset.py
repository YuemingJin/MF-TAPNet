import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from albumentations.pytorch.functional import img_to_tensor


import config


class RobotSegDataset(Dataset):
    """docstring for RobotSegDataset"""
    def __init__(self, filenames, transform):
        super(RobotSegDataset, self).__init__()
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        # num of imgs
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = load_image(filename)
        mask = load_mask(filename)
        data = {'image': image, 'mask': mask}

        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]
        
        # load optical flow
        optflow = load_optflow(filename)
        optflow = cv2.resize(optflow, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA).transpose(2,0,1)

        # TODO: change the file format
        if config.problem_type == 'binary':
            # reshape the binary output to be 3D
            return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float(), \
                        torch.from_numpy(optflow).float()
        else:
            return img_to_tensor(image), torch.from_numpy(mask).long(), \
                        torch.from_numpy(optflow).float()


def load_image(filename):
    return cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)

def load_mask(filename):
    # problem type
    problem_type = config.problem_type

    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = config.binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = config.parts_factor
    elif problem_type == 'instruments':
        factor = config.instrument_factor
        mask_folder = 'instruments_masks'

    # change dir name
    mask = cv2.imread(str(filename).replace('images', mask_folder), 0)

    return (mask / factor).astype(np.uint8)


def load_optflow(filename):
    # read .flo file
    with open(str(filename).replace('images', 'optflows').replace('png', 'flo')) as f:
        header = np.fromfile(f, dtype=np.uint8, count=4)
        size = np.fromfile(f, dtype=np.int32, count=2)
        optflow = np.fromfile(f, dtype=np.float32) \
            .reshape(config.cropped_height, config.cropped_width, 2)# .transpose(2,0,1)
    return optflow



# def test():
#     from preprocess_data import trainval_split
#     from albumentations import (
#         Compose,
#         Normalize,
#         Resize
#     )
#     train_fn, val_fn = trainval_split(0)

#     transform = Compose([
#             Resize(height=config.train_height, width=config.train_width, p=1),
#             Normalize(p=1)
#         ], p=1)

#     dataset = RobotSegDataset(train_fn, transform, train=True)
#     img, mask = dataset[0]
#     print(img.shape, mask.shape)

# if __name__ == '__main__':
#     test()
