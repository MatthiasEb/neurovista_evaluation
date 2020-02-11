import argparse
import os
from pathlib import Path
import routines
from data_generator import SupervisedGenerator
from model import nv1x16
import datetime

parser = argparse.ArgumentParser()
# Algorithm settings
parser.add_argument('-lr', '--learning_rate', help="base learning rate", type=float, default=0.001)
parser.add_argument('-b', '--batch_size', help="batchsize", type=int, default=64)
parser.add_argument('-e', '--epochs', help="epochs", type=int, default=50)
parser.add_argument('-c', '--max_channels', help="maximum number of kernels in convolution", type=int, default=256)
parser.add_argument('-gpu', '--gpu_device', help="specify, which gpu device to use for training", default=None)

# Evaluation settings
parser.add_argument('-csv', '--path', help='path to the csv that includes the files', default='.')
parser.add_argument('-m', '--mode', help='Mode. 1: training, 2: validation, 3: test', type=int, default=0,
                    choices=[0, 1, 2, 3])
parser.add_argument('-p', '--patient', help='Patient number, 1 to 15 is available', type=int, default=1)
parser.add_argument('-l', '--file_segment_length', help='Segment length in minutes, 1 or 10', type=int, default=10)
parser.add_argument('-sm', '--subtract_mean', help='Subtract channelwise mean of each segment', type=bool, default=True)


def main():
    args = parser.parse_args()
    print(args)

    if args.gpu_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    if args.mode in [0, 1]:
        routines.training(args)

if __name__ == '__main__':
    main()

