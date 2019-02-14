#!/usr/bin/env python

import torch

import getopt
import math
import numpy as np
import os
import PIL
import PIL.Image
import sys


import cv2
from albumentations.pytorch.functional import img_to_tensor
from torch.utils.data import Dataset, DataLoader
import tqdm
import config
from torch.nn.functional import interpolate
from pathlib import Path


# try:
#     from correlation import correlation # the custom cost volume layer
# except:
#     sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# # end

from correlation import correlation



Backward_tensorGrid = {}

def Backward(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
    # end

    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        class Upconv(torch.nn.Module):
            def __init__(self):
                super(Upconv, self).__init__()

                self.moduleSixOut = torch.nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=3, stride=1, padding=1)

                self.moduleSixUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

                self.moduleFivNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFivOut = torch.nn.Conv2d(in_channels=1026, out_channels=2, kernel_size=3, stride=1, padding=1)

                self.moduleFivUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

                self.moduleFouNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=1026, out_channels=256, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFouOut = torch.nn.Conv2d(in_channels=770, out_channels=2, kernel_size=3, stride=1, padding=1)

                self.moduleFouUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

                self.moduleThrNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=770, out_channels=128, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThrOut = torch.nn.Conv2d(in_channels=386, out_channels=2, kernel_size=3, stride=1, padding=1)

                self.moduleThrUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

                self.moduleTwoNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=386, out_channels=64, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwoOut = torch.nn.Conv2d(in_channels=194, out_channels=2, kernel_size=3, stride=1, padding=1)

                self.moduleUpscale = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=1, bias=False),
                    torch.nn.ReplicationPad2d(padding=[ 0, 1, 0, 1 ])
                )
            # end

            def forward(self, tensorFirst, tensorSecond, objectInput):
                objectOutput = {}

                tensorInput = objectInput['conv6']
                objectOutput['flow6'] = self.moduleSixOut(tensorInput)
                tensorInput = torch.cat([ objectInput['conv5'], self.moduleFivNext(tensorInput), self.moduleSixUp(objectOutput['flow6']) ], 1)
                objectOutput['flow5'] = self.moduleFivOut(tensorInput)
                tensorInput = torch.cat([ objectInput['conv4'], self.moduleFouNext(tensorInput), self.moduleFivUp(objectOutput['flow5']) ], 1)
                objectOutput['flow4'] = self.moduleFouOut(tensorInput)
                tensorInput = torch.cat([ objectInput['conv3'], self.moduleThrNext(tensorInput), self.moduleFouUp(objectOutput['flow4']) ], 1)
                objectOutput['flow3'] = self.moduleThrOut(tensorInput)
                tensorInput = torch.cat([ objectInput['conv2'], self.moduleTwoNext(tensorInput), self.moduleThrUp(objectOutput['flow3']) ], 1)
                objectOutput['flow2'] = self.moduleTwoOut(tensorInput)

                return self.moduleUpscale(self.moduleUpscale(objectOutput['flow2'])) * 20.0
            # end
        # end

        class Complex(torch.nn.Module):
            def __init__(self):
                super(Complex, self).__init__()

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 2, 4, 2, 4 ]),
                    torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
                    torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
                    torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleRedir = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleCorrelation = correlation.ModuleCorrelation()

                self.moduleCombined = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=473, out_channels=256, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
                    torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                
                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                
                self.moduleSix = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
                    torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleUpconv = Upconv()
            # end

            def forward(self, tensorFirst, tensorSecond, tensorFlow):
                objectOutput = {}

                assert(tensorFlow is None)

                objectOutput['conv1'] = self.moduleOne(tensorFirst)
                objectOutput['conv2'] = self.moduleTwo(objectOutput['conv1'])
                objectOutput['conv3'] = self.moduleThr(objectOutput['conv2'])

                tensorRedir = self.moduleRedir(objectOutput['conv3'])
                tensorOther = self.moduleThr(self.moduleTwo(self.moduleOne(tensorSecond)))
                tensorCorr = self.moduleCorrelation(objectOutput['conv3'], tensorOther)

                objectOutput['conv3'] = self.moduleCombined(torch.cat([ tensorRedir, tensorCorr ], 1))
                objectOutput['conv4'] = self.moduleFou(objectOutput['conv3'])
                objectOutput['conv5'] = self.moduleFiv(objectOutput['conv4'])
                objectOutput['conv6'] = self.moduleSix(objectOutput['conv5'])

                return self.moduleUpconv(tensorFirst, tensorSecond, objectOutput)
            # end
        # end

        class Simple(torch.nn.Module):
            def __init__(self):
                super(Simple, self).__init__()

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 2, 4, 2, 4 ]),
                    torch.nn.Conv2d(in_channels=14, out_channels=64, kernel_size=7, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
                    torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
                    torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
                    torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
                    torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleUpconv = Upconv()
            # end

            def forward(self, tensorFirst, tensorSecond, tensorFlow):
                objectOutput = {}

                tensorWarp = Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow)

                objectOutput['conv1'] = self.moduleOne(torch.cat([ tensorFirst, tensorSecond, tensorFlow, tensorWarp, (tensorFirst - tensorWarp).abs() ], 1))
                objectOutput['conv2'] = self.moduleTwo(objectOutput['conv1'])
                objectOutput['conv3'] = self.moduleThr(objectOutput['conv2'])
                objectOutput['conv4'] = self.moduleFou(objectOutput['conv3'])
                objectOutput['conv5'] = self.moduleFiv(objectOutput['conv4'])
                objectOutput['conv6'] = self.moduleSix(objectOutput['conv5'])

                return self.moduleUpconv(tensorFirst, tensorSecond, objectOutput)
            # end
        # end

        self.moduleFlownets = torch.nn.ModuleList([
            Complex(),
            Simple(),
            Simple()
        ])

        self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))
    # end

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = tensorFirst[:, [ 2, 1, 0 ], :, :]
        tensorSecond = tensorSecond[:, [ 2, 1, 0 ], :, :]

        tensorFirst[:, 0, :, :] = tensorFirst[:, 0, :, :] - (104.920005 / 255.0)
        tensorFirst[:, 1, :, :] = tensorFirst[:, 1, :, :] - (110.175300 / 255.0)
        tensorFirst[:, 2, :, :] = tensorFirst[:, 2, :, :] - (114.785955 / 255.0)

        tensorSecond[:, 0, :, :] = tensorSecond[:, 0, :, :] - (104.920005 / 255.0)
        tensorSecond[:, 1, :, :] = tensorSecond[:, 1, :, :] - (110.175300 / 255.0)
        tensorSecond[:, 2, :, :] = tensorSecond[:, 2, :, :] - (114.785955 / 255.0)

        tensorFlow = None

        for moduleFlownet in self.moduleFlownets:
            tensorFlow = moduleFlownet(tensorFirst, tensorSecond, tensorFlow)
        # end

        return tensorFlow
    # end
# end

##########################################################

def estimate(model, first, second):

    w, h = first.size()[2:]

    assert w % 64 == 0
    assert h % 64 == 0

    # new_w = math.ceil(w / 64.0) * 64
    # new_h = math.ceil(h / 64.0) * 64

    # assert(intWidth == 1280) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 384) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    # first = interpolate(input=first, size=(new_h, new_w), mode='bilinear', align_corners=True).cuda()
    # second = interpolate(input=second, size=(new_h, new_w), mode='bilinear', align_corners=True).cuda()

    output = model(first.cuda(), second.cuda())

    # output = interpolate(input=output, size=(h, w), mode='bilinear', align_corners=True)

    # output[:, 0, :, :] *= float(w) / float(new_w)
    # output[:, 1, :, :] *= float(h) / float(new_h)

    return output.cpu()

##########################################################



##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.cuda.device(0) # change this if you have a multiple graphics cards and you want to utilize them

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'css'


##########################################################

class RobotDataset(Dataset):
    """docstring for RobotDataset"""
    def __init__(self):
        super(RobotDataset, self).__init__()
        self.dir = config.cropped_train_dir
        self.filenames = []
        for idx in range(1, 9):
            self.filenames += list((config.cropped_train_dir / ('instrument_dataset_' + str(idx)) / 'images').glob('*'))

    def __len__(self):
        # num of imgs
        return len(self.filenames)

    def __getitem__(self, idx):
        prev_idx = idx if idx == 0 else idx - 1
        file1, file2 = self.filenames[prev_idx], self.filenames[idx]
        first = cv2.cvtColor(cv2.imread(str(file1)), cv2.COLOR_BGR2RGB)
        second = cv2.cvtColor(cv2.imread(str(file2)), cv2.COLOR_BGR2RGB)
        return str(file2), img_to_tensor(first), img_to_tensor(second)


def main():
    # first reverse the input last dimension?
    # change dimension order
    # then normalize


    model = Network().cuda().eval()

    dataset = RobotDataset()
    batch_size = 2

    loader = DataLoader(
            dataset=RobotDataset(),
            shuffle=True,
            num_workers=0,
            batch_size=batch_size,
            pin_memory=True
        )
    
    # progress bar
    tq = tqdm.tqdm(total=(len(loader) * batch_size))
    tq.set_description('get optical flow')
    
    for i, (filenames, firsts, seconds) in enumerate(loader):
        outputs = estimate(model, firsts, seconds)

        for filename, output in zip(filenames, outputs):
            # save as .flo format
            optfilename = Path(filename.replace('images', 'optflows').replace('png', 'flo'))
            optfilename.parent.mkdir(exist_ok=True, parents=True)

            objectOutput = open(str(optfilename), 'wb')

            # save as .flo format
            np.array([80, 73, 69, 72], np.uint8).tofile(objectOutput)
            np.array([output.size(2), output.size(1)], np.int32).tofile(objectOutput)
            np.array(output.numpy().transpose(1, 2, 0), np.float32).tofile(objectOutput)

            objectOutput.close()

        tq.update(batch_size)


def test_file():
    model = Network().cuda().eval()
    fn1 = 'pytorch-unflow-master/images/first.png'
    fn2 = 'pytorch-unflow-master/images/second.png'
    first = img_to_tensor(cv2.cvtColor(cv2.imread(str(fn1)), cv2.COLOR_BGR2RGB))
    second = img_to_tensor(cv2.cvtColor(cv2.imread(str(fn2)), cv2.COLOR_BGR2RGB))
    output = estimate(model, first.unsqueeze(0), second.unsqueeze(0)).squeeze()
    objectOutput = open(str('out.flo'), 'wb')

    np.array([80, 73, 69, 72], np.uint8).tofile(objectOutput)
    np.array([output.size(2), output.size(1)], np.int32).tofile(objectOutput)
    np.array(output.numpy().transpose(1, 2, 0), np.float32).tofile(objectOutput)
    out = output.numpy().transpose(1, 2, 0)
    print(out.shape, out[0])

    objectOutput.close()
    return out
    

def test_read():
    filename = config.cropped_train_dir / 'instrument_dataset_1' / 'optflows' / 'frame000.flo'
    with open(str(filename), 'rb') as f:
        header = np.fromfile(f, dtype=np.uint8, count=4)
        size = np.fromfile(f, dtype=np.int32, count=2)
        optflow = np.fromfile(f, dtype=np.float32).reshape(config.cropped_height, config.cropped_width, 2)
        optflow = torch.from_numpy(optflow.transpose(2,0,1)).float()
        return optflow


if __name__ == '__main__':
    # main()
    test_read()
    # test_file()
