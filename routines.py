from pathlib import Path
from data_generator import TrainingGenerator, EvaluationGenerator
from model import nv1x16
import datetime, os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score


def get_csv(args):
    if args.run_on_contest_data:
        csv = {
            1: 'contest_train_data_labels.csv',
            3: 'contest_test_data_labels_public.csv'
        }
    else:
        csv = {
            1: 'train_filenames_labels_patient{}_segment_length_{}.csv'.format(args.pat, args.file_segment_length),
            2: 'validation_filenames_patient{}_segment_length_{}.csv'.format(args.pat, args.file_segment_length),
            3: 'test_filenames_patient{}_segment_length_{}.csv'.format(args.pat, args.file_segment_length)
        }

    return Path(args.CSV) / csv[args.mode]


def training(args):
    if not os.path.isfile(get_csv(args)):
        raise FileNotFoundError(
            'CSV file not found: {}'.format(get_csv(args)))

    if args.subtract_mean and not args.run_on_contest_data:
        standardize_mode = 'file_channelwise'
    else:
        standardize_mode = None

    df_filenames = pd.read_csv(get_csv(args))

    if args.run_on_contest_data:
        train_runs = [(df_filenames.loc[df_filenames['image'].str.contains('Pat{}'.format(i))], i) for i in range(1,4)]
        args.segment_length_minutes = 10
    else:
        train_runs = [(df_filenames, args.pat)]

    for df_filenames, patient in train_runs:
        print('Starting Training Patient {}...'.format(patient))
        dg = TrainingGenerator(df_filenames_csv=df_filenames,
                               file_segment_length=args.segment_length_minutes,
                               buffer_length=4000,
                               batch_size=40,
                               n_workers=5,
                               standardize_mode=standardize_mode,
                               shuffle=True)

        model = nv1x16()

        model.summary()
        history = model.fit(x=dg,
                            shuffle=False,  # do not change these settings!
                            use_multiprocessing=False,
                            verbose=1,
                            workers=0,
                            epochs=1)
        print('training patient {} done'.format(patient))
        Path(args.model).mkdir(exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        pat = 'pat{}'.format(patient)
        if args.run_on_contest_data:
            sl = sm = ''
        else:
            sl = '_{}min'.format(args.file_segment_length)
            sm = '_sub{}'.format(args.subtract_mean)
        model_file = '{}{}{}_model_weights.h5'.format(pat, sl, sm)
        settings = os.path.join(args.model, '{}_{}_settings.txt'.format(current_time, pat))
        logs = os.path.join(args.model, '{}_{}_logs.csv'.format(current_time, pat))
        model_archive = os.path.join(args.model, model_file)
        print('Archiving model weights to ' + model_archive)
        model.save_weights(model_archive)
        print('Archiving settings to ' + settings)
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



    df_filenames = pd.read_csv(get_csv(args))

    if args.run_on_contest_data:
        dfs = [(df_filenames.loc[df_filenames['image'].str.startswith('Pat{}'.format(i))], i) for i in range(1,4)]

    else:
        dfs = [(df_filenames, args.pat)]
    for df_filenames, patient in dfs:
        print('Starting Evaluation for Patient {}...'.format(patient))

        pat = 'pat{}'.format(patient)
        if args.run_on_contest_data:
            sl = sm = ''
        else:
            sl = '_{}min'.format(args.file_segment_length)
            sm = '_sub{}'.format(args.subtract_mean)
        model_file = '{}{}{}_model_weights.h5'.format(pat, sl, sm)

        model = nv1x16()
        model_path = os.path.join(args.model, model_file)

        try:
            model.load_weights(model_path)
        except FileNotFoundError:
            raise FileNotFoundError('Please train model for specified options and patient before evaluating.')

        dg = EvaluationGenerator(df_filenames_csv=df_filenames,
                                 file_segment_length=args.file_segment_length,
                                 standardize_mode=standardize_mode,
                                 batch_size=args.file_segment_length * 4,
                                 class_weights=None)

        probs = model.predict(dg, verbose=args.verbose)
        df = pd.read_csv(get_csv(args), index_col=0)
        probs = probs.reshape(len(df), -1).mean(axis=1)
        if args.mode == 1:
            print('Results on training set:')
            metrics = [roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score]
            names = ['roc_auc', 'average_precision', 'precision', 'recall', 'accuracy']

            for m, n in zip(metrics[:2], names[:2]):
                print('{}: {:.4f}'.format(n, m(df['class'], probs)))
            print('For Threshold = 0.5:')
            for m, n in zip(metrics[2:], names[2:]):
                print('{}: {:.4f}'.format(n, m(df['class'], probs > 0.5)))

        df['class'] = probs

        Path('solutions').mkdir(exist_ok=True)
        fn = 'solution_matthiasEb_pat{}_seg{}_mode{}_subtract{}.csv'.format(args.pat,
                                                                            args.file_segment_length,
                                                                            args.mode,
                                                                            args.subtract_mean)

        df.to_csv('solutions/' + fn)

    print('Evaluation done')
