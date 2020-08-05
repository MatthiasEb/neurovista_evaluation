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

    print(json_dict)
    args = Namespace(json_dict)

    # run training
    if args.mode == 1:
        routines.training(args)
        if args.run_on_contest_data:
            routines.evaluate(args)
    else:
        routines.evaluate(args)


if __name__ == '__main__':
    main()
