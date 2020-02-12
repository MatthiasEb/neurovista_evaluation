import argparse
import os
import routines
import warnings

parser = argparse.ArgumentParser()
# Algorithm settings
parser.add_argument('-b', '--batch_size', help="batchsize", type=int, default=80)
parser.add_argument('-e', '--epochs', help="training epochs", type=int, default=50)
parser.add_argument('-v', '--verbose', help="tf.keras verbosity", type=int, default=1, choices=[0, 1, 2])
parser.add_argument('-c', '--max_channels', help="maximum number of kernels in convolution", type=int, default=128)
parser.add_argument('-gpu', '--gpu_device', help="Which gpu device to use. If unspecified, use CPU.", default=None)
parser.add_argument('-mf', '--model_file', help="Path to stored model file for model evaluation. "
                                                "If not specified, trained model for respecitve pateint is expected in "
                                                "current working directory", default=None)

# Evaluation settings
parser.add_argument('-csv', '--path', help='path to the csv that includes the files', default='.')
parser.add_argument('-m', '--mode', help='Mode. 1: training, 2: validation, 3: test', type=int, default=0,
                    choices=[0, 1, 2, 3])
parser.add_argument('-p', '--patient', help='Patient number, 1 to 15 is available', type=int, default=1)
parser.add_argument('-l', '--file_segment_length', help='Segment length in minutes, 1 or 10', type=int, default=10)
parser.add_argument('-sm', '--subtract_mean', help='Subtract channelwise mean of each file', type=int, default=1,
                    choices=[0, 1])


def main():
    args = parser.parse_args()
    print(args)

    # setup devices to use, throw warnings if settings seem to be unintended
    if args.gpu_device is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        if args.mode == 1:
            warnings.warn('Training without GPU!')
    elif args.gpu_device is 'all':
        pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
        if args.mode in [2, 3]:
            warnings.warn('Validating/Testing with GPU!')

    # run training
    if args.mode in [0, 1]:
        routines.training(args)

    routines.evaluate(args)


if __name__ == '__main__':
    main()
