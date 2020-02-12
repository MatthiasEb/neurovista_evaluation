from pathlib import Path
from data_generator import TrainingGenerator, EvaluationGenerator
from model import nv1x16
import datetime, os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score


def get_csv(args):
    csv = {
        0: 'contest_train_data_labels.csv',
        1: 'train_filenames_labels_patient{}_segment_length_{}.csv'.format(args.patient, args.file_segment_length),
        2: 'validation_filenames_patient{}_segment_length_{}.csv'.format(args.patient, args.file_segment_length),
        3: 'test_filenames_patient{}_segment_length_{}.csv'.format(args.patient, args.file_segment_length)
    }

    return Path(args.path) / csv[args.mode]


def training(args):
    print('Starting Training...')
    if not os.path.isfile(Path(args.path) / get_csv(args)):
        raise FileNotFoundError(
            'Please specify the path where the csv file can be found or copy the csv file to current location')

    if args.subtract_mean:
        standardize_mode = 'file_channelwise'
    else:
        standardize_mode = None

    dg = TrainingGenerator(filenames_csv_path=Path(args.path) / get_csv(args),
                           file_segment_length=args.file_segment_length,
                           buffer_length=4000,
                           batch_size=args.batch_size,
                           n_workers=10,
                           standardize_mode=standardize_mode,
                           shuffle=True)

    model = nv1x16(args.max_channels)

    model.summary()
    history = model.fit(x=dg,
                        shuffle=False,  # do not change these settings!
                        use_multiprocessing=False,
                        verbose=args.verbose,
                        workers=0,
                        epochs=args.epochs)
    print('training done')
    Path('archive').mkdir(exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    pat = 'pat{}'.format(args.patient)
    sl = '{}min'.format(args.file_segment_length)
    sm = 'sub{}'.format(args.subtract_mean)
    model_file = '{}_{}_{}_model_weights.h5'.format(pat, sl, sm)
    settings = 'archive/{}_{}_settings.txt'.format(current_time, pat)
    logs = 'archive/{}_{}_logs.csv'.format(current_time, pat)
    model_archive = 'archive/{}_{}_{}_{}_model_weights.h5'.format(current_time, pat, sl, sm)
    print('Saving model weights to ' + model_file)
    model.save_weights(model_file)
    print('Archiving model weights to ' + model_archive)
    model.save_weights(model_archive)
    print('Archiving args to ' + settings)
    with open(settings, 'w') as s:
        s.write(str(args))
    print('Archiving training history to ' + logs)
    pd.DataFrame(history.history).to_csv(logs)


def evaluate(args):
    print('Starting Evaluation...')
    if args.subtract_mean:
        standardize_mode = 'file_channelwise'
    else:
        standardize_mode = None

    model = nv1x16(args.max_channels)
    if args.model_file:
        model.load_weights(args.model_file)
    else:
        model.load_weights('pat{}_{}min_sub{}_model_weights.h5'.format(args.patient,
                                                                       args.file_segment_length, args.subtract_mean))

    dg = EvaluationGenerator(filenames_csv_path=get_csv(args),
                             file_segment_length=args.file_segment_length,
                             standardize_mode=standardize_mode,
                             batch_size=args.file_segment_length * 4,
                             )
    probs = model.predict(dg, verbose=args.verbose)
    df = pd.read_csv(get_csv(args), index_col=0)
    probs = probs.reshape(len(df), -1).mean(axis=1)
    if args.mode in [0, 1]:
        print('Results on training set:')
        metrics = [roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score]
        names = ['roc_auc', 'average_precision', 'precision', 'recall', 'accuracy']

        for m, n in zip(metrics[:2], names[:2]):
            print('{}: {:.4f}'.format(n, m(df['class'], probs)))
        print('For Threshold = 0.5:')
        for m, n in zip(metrics[2:], names[2:]):
            print('{}: {:.4f}'.format(n, m(df['class'], probs>0.5)))

    df['class'] = probs

    Path('solutions').mkdir(exist_ok=True)
    fn = 'solution_matthiasEb_pat{}_seg{}_mode{}_subtract{}.csv'.format(args.patient,
                                                                        args.file_segment_length,
                                                                        args.mode,
                                                                        args.subtract_mean)

    df.to_csv('solutions/' + fn)

    print('Evaluation done')
