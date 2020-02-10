# %% import statements

import tensorflow as tf
import numpy as np
import scipy.io as io
import pandas as pd

# define several possible standardization methods
STANDARDIZE_OPTIONS = ['global_channelwise', 'global', 'segment_channelwise', 'segment', 'batch', 'batch_channelwise',
                       'file', 'file_channelwise', None]
# standardization methods that use the same scalings for the whole set
GLOBAL_STANDARDIZE = STANDARDIZE_OPTIONS[:2]
# generator running in train/val/test mode?
MODE_OPTIONS = [0, 1, 2, 3]
# generator returns labels only in train mode
RETURN_LABELS = MODE_OPTIONS[:1]


# %% define Tensorflow Generator
class SegmentGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 filenames_csv_path,
                 file_segment_length,
                 buffer_length=4000,
                 expand_cnn_dim=True,
                 batch_size=1,
                 shuffle=True,
                 standardize_mode=None,
                 mode=0,
                 mean=None,
                 std=None,
                 class_weights='auto'):

        # --- pass arguments ---
        self.batch_size = batch_size
        self.file_segment_length = file_segment_length
        self.expand_cnn_dim = expand_cnn_dim
        self.csv = pd.read_csv(filenames_csv_path)
        if mode not in MODE_OPTIONS:
            raise ValueError("'mode' hast to be one of {}.".format(MODE_OPTIONS))
        self.mode = mode
        self.segm_length = 6000
        self.n_channels = 16
        self.samples_per_file = self.file_segment_length * 4  # draw 15 s segments
        if standardize_mode not in STANDARDIZE_OPTIONS:
            raise ValueError("'standardize_mode' hast to be one of {}.".format(STANDARDIZE_OPTIONS))

        self.shape = (len(self.csv)*self.samples_per_file, self.segm_length, self.n_channels)
        if self.expand_cnn_dim:
            self.shape += (1,)
        self.shuffle = shuffle

        # --- setup class weights ---
        if self.mode not in RETURN_LABELS and class_weights is not None:
            raise RuntimeWarning("in validation and test modes, class weights are ignored.")

        if class_weights == 'auto':
            pi = self.csv['class'].sum()
            ii = len(self.csv) - pi
            self.class_weights = {0: 1, 1: ii / pi}

            print('Class weights were set automatically based on the ratio in the training data')
            print('II: {}, PI: {}'.format(self.class_weights[0], self.class_weights[1]))

        elif class_weights is not None:
            self.class_weights = class_weights
        else:
            self.class_weights = None

        # --- setup file IO ---
        # open a specified number of files, draw samples randomly from those opened files
        self.buffer_length = buffer_length
        self.buffer = []
        self.n_file = len(self.csv)


        # --- setup standardization ---
        self.standardize = None
        self.on_epoch_end()
        self.norm_count = 0
        # optional: standardize train set with global mean/std
        if standardize_mode in GLOBAL_STANDARDIZE:
            if mode in [0, 1]:
                print('Calculating Train Mean, Train STD...')
                mean = 0
                std = 0

                for data in self:
                    data = data[0]
                    if standardize_mode == 'global_channelwise':
                        mean += data.mean(axis=1).mean(axis=0).squeeze()
                        std += data.std(axis=1).mean(axis=0).squeeze()
                    else:
                        mean += data.mean()
                        std += data.std()

                mean = mean / len(self)
                std = std / len(self)
            # if train set was standardized by global mean/std , use train sets' mean/std to standardize test/validation set
            elif mean is None or std is None:
                raise ValueError('Need Training Mean and STD as argument for standardizing validation or test data')

        self.mean = mean
        self.std = std
        self.standardize = standardize_mode

    def fill_buffer(self):
        # check, if not all files have been loaded already

        while len(self.buffer) < self.buffer_length and self.n_file < len(self.csv):
            m = io.loadmat(self.csv.iloc[self.n_file]['image'])

            x = m['dataStruct'][0][0][0]
            if self.standardize == 'file':
                x -= x.mean()
                x /= x.std()
            elif self.standardize == 'file_channelwise':
                x -= x.mean(axis=0)
                x /= x.std(axis=0)
            x = x.reshape(-1, 1, self.segm_length, self.n_channels).astype('float32')
            if self.mode in RETURN_LABELS:
                y = np.ones(x.shape[0]) * self.csv.iloc[self.n_file]['class']
                self.buffer += list(zip(x, y))
            else:
                self.buffer += list(x)
            self.n_file += 1

        if self.shuffle:
            np.random.shuffle(self.buffer)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            idx = np.arange(len(self.csv))
            np.random.shuffle(idx)
            self.csv = self.csv.iloc[idx]
        # check, if we have run through the whole dataset
        assert self.n_file == len(self.csv)
        assert len(self.buffer) <= self.batch_size
        self.n_file = 0

    def __len__(self):
        return int(np.ceil(len(self.csv)*self.samples_per_file / self.batch_size))

    def __getitem__(self, index):
        """
        Generates one batch of data.

        :param index: Index
        :type index: int
        :return: Batch
        """
        if self.n_file < len(self.csv):
            self.fill_buffer()  # fill buffer
        last_batch = len(self.buffer) <= self.batch_size
        return_labels = self.mode in RETURN_LABELS

        # Pop batch
        if not last_batch:
            b = np.array(self.buffer[:self.batch_size])
            del self.buffer[:self.batch_size]

        else:
            b = np.array(self.buffer[:])
            # do not delete last batch, since queue will be filled with it

        if return_labels:
            x = np.concatenate(b[:, 0])
        else:
            x = np.concatenate(b)

        if self.standardize is not None:
            x = self.norm_scale(x)

        if self.expand_cnn_dim:
            x = np.expand_dims(x, -1)
        if return_labels:
            y = np.array(b[:, 1], dtype=int)
            if self.class_weights is None:
                return x, y
            else:
                sw = np.zeros(y.shape)
                sw[y == 0] = self.class_weights[0]
                sw[y == 1] = self.class_weights[1]
                return x, y, sw
        else:
            return x

    def norm_scale(self, data):
        if self.standardize in ['file', 'file_channelwise']:
            return data
        if self.standardize in GLOBAL_STANDARDIZE:
            pass
        elif self.standardize == 'batch':
            self.mean = data.mean()
            self.std = data.std()
        elif self.standardize == 'batch_channelwise':
            self.mean = data.mean(axis=(0, 1))
            self.std = data.std(axis=(0, 1)) + np.finfo(np.float32).eps
        elif self.standardize == 'segment_channelwise':
            self.mean = data.mean(axis=1)[:, np.newaxis]
            self.std = data.std(axis=1)[:, np.newaxis] + np.finfo(np.float32).eps
        elif self.standardize == 'segment':
            self.mean = data.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]
            self.std = data.std(axis=(1, 2))[:, np.newaxis, np.newaxis] + np.finfo(np.float32).eps

        return (data - self.mean) / self.std


# %% execute only if run as a script
if __name__ == '__main__':
    # %% provide inputs
    fn ='/home/s4238870/code/neurovista_evaluation/sample.csv'
    sl = 10  # segment length in minutes
    test_normalize = False
    test_epoch = False
    test_data = True
    # %% instantiate SegmentGenerator
    bs = 40
    if test_normalize:
        for run, sm in enumerate(STANDARDIZE_OPTIONS):
            print('\n\n=== Standardization mode: {} ==='.format(sm))
            gen = SegmentGenerator(fn,
                                   sl,
                                   mode=0,
                                   shuffle=True,
                                   standardize_mode=sm,
                                   batch_size=bs)
            k = []
            batch = np.empty(1)
            if run == 0:
                print('Generating 10 batches...')
            for i in range(len(gen)):
                batch = gen[i]
                k.append(batch)
            if run == 0:
                print('First Batch shape: {}'.format(k[0][0].shape))
                print('Last Batch shape: {}'.format(k[-1][0].shape))
            X = np.concatenate([x[0] for x in k], axis=0)
            if run == 0:
                print('Data shape: {}'.format(X.shape))
            X = X.squeeze()
            print('\n----channelwise----')
            print('STD: {}'.format(X.std(-2).mean(axis=tuple([i for i in range(X.ndim - 2)]))))
            print('MEAN: {}'.format(X.mean(-2).mean(axis=tuple([i for i in range(X.ndim - 2)]))))

            print('\n----total----')
            print('STD: {}'.format(X.std(-2).mean()))
            print('MEAN: {}'.format(X.mean()))

    if test_epoch:
        gen = SegmentGenerator(fn,
                               sl,
                               mode=0,
                               shuffle=True,
                               standardize_mode='file_channelwise',
                               batch_size=bs)
        for epoch in range(10):
            print('Epoch: {}'.format(epoch))
            for i in range(len(gen)):
                X = gen[i]
            gen.on_epoch_end()

    if test_data:
        gen = SegmentGenerator(fn,
                               sl,
                               mode=0,
                               shuffle=False,
                               standardize_mode=None,
                               batch_size=bs)
        for i in range(len(gen)):
            x, y, z = gen[i]
            m = io.loadmat(gen.csv.iloc[i]['image'])['dataStruct'][0][0][0]
            assert np.array_equal(x.reshape(m.shape), m)

    print('test passed.')