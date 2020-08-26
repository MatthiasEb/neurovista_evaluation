from pathlib import Path
from data_generator import TrainingGenerator, EvaluationGenerator
from model import nv1x16
import os
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
            1: 'train_filenames_labels_patient{}_segment_length_{}.csv'.format(args.pat, args.segment_length_minutes),
            2: 'validation_filenames_patient{}_segment_length_{}.csv'.format(args.pat, args.segment_length_minutes),
            3: 'test_filenames_patient{}_segment_length_{}.csv'.format(args.pat, args.segment_length_minutes)
        }

    return Path(args.CSV) / csv[args.mode]


def load_csv(args):
    csv = pd.read_csv(get_csv(args))
    try:
        _ = csv['image']
    except KeyError:
        csv = csv.rename(columns={'image ': 'image'})  # handle inconsistent column naming in csv
        print('renamed column')
    # --- check, if data specified in csv are existent ---
    for i, f in enumerate(csv['image']):
        if not os.path.isfile(f):
            # try path relative to csv as in the sample files
            if os.path.isfile(os.path.join(args.CSV, f)):
                csv.loc[i, 'image'] = os.path.join(args.CSV, f)
            else:
                raise FileNotFoundError('File {} from csv does not exist.'.format(f))

    return csv


def training(args):
    if not os.path.isfile(get_csv(args)):
        raise FileNotFoundError(
            'CSV file not found: {}'.format(get_csv(args)))

    if args.subtract_mean:
        standardize_mode = 'file_channelwise'
    else:
        standardize_mode = None

    df_filenames = load_csv(args)

    if args.run_on_contest_data:
        train_runs = [(df_filenames.loc[df_filenames['image'].str.contains('Pat{}'.format(i))], i) for i in range(1, 4)]
    else:
        train_runs = [(df_filenames, args.pat)]

    for df_filenames, patient in train_runs:
        print('Starting Training Patient {}...'.format(patient))
        dg = TrainingGenerator(df_filenames_csv=df_filenames,
                               segment_length_minutes=args.segment_length_minutes,
                               buffer_length=4000,
                               batch_size=40,
                               n_workers=5,
                               standardize_mode=standardize_mode,
                               shuffle=True)

        model = nv1x16()

        model.fit(x=dg,
                  shuffle=False,  # do not change these settings!
                  use_multiprocessing=False,
                  verbose=2,
                  workers=1,
                  epochs=50)
        print('training patient {} done'.format(patient))
        Path(args.model).mkdir(exist_ok=True)
        model_file = 'model_dataset{}_pat{}_seg{}_subtract{}.h5'.format(args.run_on_contest_data,
                                                                        patient,
                                                                        args.segment_length_minutes,
                                                                        args.subtract_mean)
        model_archive = os.path.join(args.model, model_file)
        print('Archiving model weights to ' + model_archive)
        model.save_weights(model_archive)


def evaluate(args):
    print('Starting Evaluation...')
    if args.subtract_mean:
        standardize_mode = 'file_channelwise'
    else:
        standardize_mode = None

    df_filenames = load_csv(args)

    if args.run_on_contest_data:
        dfs = [(df_filenames.loc[df_filenames['image'].str.contains('Pat{}'.format(i))], i) for i in range(1, 4)]
    else:
        dfs = [(df_filenames, args.pat)]

    solutions = []
    for df_filenames, patient in dfs:
        print('Starting Evaluation for Patient {}...'.format(patient))

        model_file = 'model_dataset{}_pat{}_seg{}_subtract{}.h5'.format(args.run_on_contest_data,
                                                                        patient,
                                                                        args.segment_length_minutes,
                                                                        args.subtract_mean)

        model = nv1x16()
        model_path = os.path.join(args.model, model_file)
        print('Using model located at: {}'.format(model_path))

        try:
            model.load_weights(model_path)
        except FileNotFoundError:
            raise FileNotFoundError('Please train model for specified options and patient before evaluating.')

        dg = EvaluationGenerator(df_filenames_csv=df_filenames,
                                 segment_length_minutes=args.segment_length_minutes,
                                 standardize_mode=standardize_mode,
                                 batch_size=args.segment_length_minutes * 4,
                                 class_weights=None)

        probs = model.predict(dg, verbose=0)

        probs = probs.reshape(len(df_filenames), -1).mean(axis=1)
        if args.mode == 1:
            print('Results on training set:')
            metrics = [roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score]
            names = ['roc_auc', 'average_precision', 'precision', 'recall', 'accuracy']

            for m, n in zip(metrics[:2], names[:2]):
                print('{}: {:.4f}'.format(n, m(df_filenames['class'], probs)))
            print('For Threshold = 0.5:')
            for m, n in zip(metrics[2:], names[2:]):
                print('{}: {:.4f}'.format(n, m(df_filenames['class'], probs > 0.5)))

        df_filenames['class'] = probs

        solutions.append(df_filenames)

    s = pd.concat(solutions)
    s['image'] = s['image'].str.split('/').str[-1]
    s['image'] = s['image'].str.split('.').str[0]
    s = s[['image', 'class']]

    Path(args.solutions).mkdir(exist_ok=True)
    if args.run_on_contest_data:
        fn = 'contest_data_solution_matthiasEb_mode{}.csv'.format(args.mode)
    else:
        fn = 'solution_matthiasEb_pat{}_seg{}_mode{}_subtract{}.csv'.format(args.pat,
                                                                            args.segment_length_minutes,
                                                                            args.mode,
                                                                            args.subtract_mean)
    s.to_csv(os.path.join(args.solutions, fn), index=False)

    print('Saving solution file to : {}'.format(os.path.join(args.solutions, fn)))
    print('Evaluation done')
