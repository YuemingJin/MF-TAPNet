"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np
import config

# set folds[fold] as validation set
# others as training_set
def trainval_split(fold):

    # self defined train_val split
    folds = {0: [1, 3],
             1: [2, 5],
             2: [4, 8],
             3: [6, 7]}

    train_file_names = []
    val_file_names = []

    for idx in range(1, 9):
        filenames = list((config.cropped_train_dir / ('instrument_dataset_' + str(idx)) / 'images').glob('*'))
        if idx in folds[fold]:
            val_file_names += filenames
        else:
            train_file_names += filenames

    return train_file_names, val_file_names

def preprocess_data():
    cropped_train_dir = config.cropped_train_dir
    cropped_train_dir.mkdir(exist_ok=True, parents=True)

    for idx in range(1, 9):
        cropped_instrument_folder = cropped_train_dir / ('instrument_dataset_' + str(idx))

        # mkdir for each datatype
        image_folder = cropped_instrument_folder / 'images'
        image_folder.mkdir(exist_ok=True, parents=True)

        binary_mask_folder = cropped_instrument_folder / 'binary_masks'
        binary_mask_folder.mkdir(exist_ok=True, parents=True)

        parts_mask_folder = cropped_instrument_folder / 'parts_masks'
        parts_mask_folder.mkdir(exist_ok=True, parents=True)

        instrument_mask_folder = cropped_instrument_folder / 'instruments_masks'
        instrument_mask_folder.mkdir(exist_ok=True, parents=True)

        # original dataset dir
        instrument_folder = config.train_dir / ('instrument_dataset_' + str(idx))
        
        # only read left frames
        # crop (height, width) frames from (h_start, w_start)
        h_start, w_start = config.h_start, config.w_start
        cropped_height, cropped_width = config.cropped_height, config.cropped_width


        # mask folder
        mask_folders = list((instrument_folder / 'ground_truth').glob('*'))

        # frames dir
        frames_dir = instrument_folder / 'left_frames'
        for file_name in tqdm(list(frames_dir.glob('*'))):
            img = cv2.imread(str(file_name))
            old_h, old_w, _ = img.shape

            img = img[h_start: h_start + cropped_height, w_start: w_start + cropped_width]
            # save cropped image
            # cv2.imwrite(str(image_folder / (file_name.stem + '.jpg')), img,
            #             [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(str(image_folder / (file_name.name)), img)

            # save different masks
            mask_binary = np.zeros((old_h, old_w))
            mask_parts = np.zeros((old_h, old_w))
            mask_instruments = np.zeros((old_h, old_w))

            for mask_folder in mask_folders:
                # read in grayscale
                mask = cv2.imread(str(mask_folder / file_name.name), 0)

                # mark each type of instruments
                # background will be set to 0 in default
                if 'Bipolar_Forceps' in str(mask_folder):
                    mask_instruments[mask > 0] = 1
                elif 'Prograsp_Forceps' in str(mask_folder):
                    mask_instruments[mask > 0] = 2
                elif 'Large_Needle_Driver' in str(mask_folder):
                    mask_instruments[mask > 0] = 3
                elif 'Vessel_Sealer' in str(mask_folder):
                    mask_instruments[mask > 0] = 4
                elif 'Grasping_Retractor' in str(mask_folder):
                    mask_instruments[mask > 0] = 5
                elif 'Monopolar_Curved_Scissors' in str(mask_folder):
                    mask_instruments[mask > 0] = 6
                elif 'Other' in str(mask_folder):
                    mask_instruments[mask > 0] = 7

                # process dir exclude 'Other_labels'
                if 'Other' not in str(mask_folder):
                    # if exists, will be add in
                    mask_binary += mask

                    # different parts
                    mask_parts[mask == 10] = 1  # Shaft
                    mask_parts[mask == 20] = 2  # Wrist
                    mask_parts[mask == 30] = 3  # Claspers

            mask_binary = (mask_binary[h_start: h_start + cropped_height, w_start: w_start + cropped_width] > 0).astype(
                np.uint8) * config.binary_factor
            mask_parts = (mask_parts[h_start: h_start + cropped_height, w_start: w_start + cropped_width]).astype(
                np.uint8) * config.parts_factor
            mask_instruments = (mask_instruments[h_start: h_start + cropped_height, w_start: w_start + cropped_width]).astype(
                np.uint8) * config.instrument_factor

            cv2.imwrite(str(binary_mask_folder / file_name.name), mask_binary)
            cv2.imwrite(str(parts_mask_folder / file_name.name), mask_parts)
            cv2.imwrite(str(instrument_mask_folder / file_name.name), mask_instruments)


if __name__ == '__main__':
    preprocess_data()
