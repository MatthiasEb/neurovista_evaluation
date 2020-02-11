from pathlib import Path
from data_generator import SupervisedGenerator
from model import nv1x16
import datetime, os
import pandas as pd


def training(args):
    csv = {
        0: 'sample.csv',
        1: 'train_filenames_labels_patient{}_segment_length_{}.csv'.format(args.patient, args.file_segment_length),
        2: 'validation_filenames_patient{}_segment_length_{}.csv'.format(args.patient, args.file_segment_length),
        3: 'test_filenames_patient{}_segment_length_{}.csv'.format(args.patient, args.file_segment_length)
    }

    if not os.path.isfile(Path(args.path) / csv[args.mode]):
        raise FileNotFoundError(
            'Please specify the path where the csv file can be found or copy the csv file to current location')

    dg = SupervisedGenerator(filenames_csv_path=Path(args.path) / csv[args.mode],
                             file_segment_length=args.file_segment_length,
                             buffer_length=4000,
                             batch_size=80,
                             n_workers=10,
                             standardize_mode='file_channelwise',
                             shuffle=True)

    model = nv1x16()

    model.summary()
    history = model.fit(x=dg,
                        shuffle=False,  # do not change these settings!
                        use_multiprocessing=False,
                        workers=1,
                        epochs=1)
    print('training done')
    Path('archive').mkdir(exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    pat = 'pat{}'.format(args.patient)
    model_file = '{}_model.h5'.format(pat)
    settings = 'archive/{}_{}_settings.txt'.format(current_time, pat)
    logs = 'archive/{}_{}_logs.csv'.format(current_time, pat)
    model_archive = 'archive/{}_{}_model.h5'.format(current_time, pat)

    model.save(model_file)
    model.save(model_archive)
    with open(settings, 'w') as s:
        s.write(str(args))
    pd.DataFrame(history.history).to_csv(logs)
