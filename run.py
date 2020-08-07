import os
import routines
import warnings
import json

class Namespace(object):
    def __init__(self, a_dict):
        self.__dict__.update(a_dict)

    def __repr__(self):
        return self.__dict__.__repr__()

def main():
    with open('SETTINGS.json') as f:
        json_dict = json.load(f)

    args = Namespace(json_dict)

    if args.run_on_contest_data:
        args.segment_length_minutes = 10
        args.subtract_mean = 1

    print(args)

    # run training
    if args.mode == 1:
        routines.training(args)
        routines.evaluate(args)
    else:
        routines.evaluate(args)


if __name__ == '__main__':
    main()
