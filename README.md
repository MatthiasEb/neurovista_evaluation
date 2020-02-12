# neurovista_evaluation
code for evaluation on full neurovista trial data according to specifications from 6.2.2020.

## requirements
1. python3
2. cuda toolbox 10, nvidia-drivers

## installation
Install requirements by running.

`pip install -r requirements.txt`

I'd recommend using a virtual environment for this.

## execution
For detailed options, run:

`python run.py -h`

The .csv files containing the paths of the segments (and targets for training) are expected to be in 
the current working directory if not specified by the argument

`--path PATH_TO_CSV`

For approximate reproducing the results from the paper (not exactly possible due to different 
training process) execute:

`python run.py -gpu GPU_DEVICE --mode 1 -pat PATIENT`

to train the model. Warning: Displayed metrics that cannot be calculated accumulatively (roc_auc, 
precision, recall, pr_auc) are only batch-wise approximations. 

Depending on Hardware, you may want or have to change batchsize by specifying argument

`--batch_size BATCH_SIZE`

Then you can validate it with

`python run.py --mode 2 --patient PATIENT`

Testing works accordingliy.

Validation/Testing can also be done on gpu if you specify a device. I did not yet constrain the
implementation to run only single-threaded or measure maximum ram used during evaluation. 

All models, training logs and settings are saved in the `archive` directory. If you do not specify any 
model when testing, last model that was trained with the given evaluation settings (segment length, 
subtract_mean, patient) will be evaluated.

Since I could not test it in the original setup with the original csv files and data, it is possible
that there are bugs that I did not think of at this point.

Do not hesitate to contact me if you have any questions or remarks.



