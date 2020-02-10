import argparse
import os
from pathlib import Path
from data_generator import SegmentGenerator
from model import nv1x16
import xarray as xr

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

args = parser.parse_args()

print(args)

if args.gpu_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

csv = {
    0: 'contest_train_data_labels.csv',
    1: 'train_filenames_labels_patient{}_segment_length_{}.csv'.format(args.patient, args.file_segment_length),
    2: 'validation_filenames_patient{}_segment_length_{}.csv'.format(args.patient, args.file_segment_length),
    3: 'test_filenames_patient{}_segment_length_{}.csv'.format(args.patient, args.file_segment_length)
}

if not os.path.isfile(Path(args.path) / csv[args.mode]):
    raise FileNotFoundError(
        'Please specify the path where the csv file can be found or copy the csv file to current location')

dg = SegmentGenerator(filenames_csv_path=Path(args.path) / csv[args.mode],
                      file_segment_length=args.file_segment_length,
                      batch_size=40)

model = nv1x16()

model.summary()
if args.mode in [0, 1]:
    model.fit(x=dg, shuffle=False, use_multiprocessing=True, workers=6, epochs=10)

print('done')
