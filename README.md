# neurovista_evaluation
code for evaluation on full neurovista trial data according to specifications from 6.2.2020.

## Using Python
### requirements
1. python3
2. cuda toolbox 10, nvidia-drivers
 
### installation
Install requirements by running:

`pip install -r requirements.txt`

I'd recommend using a virtual environment for this.

### execution
Run training by executing:

`python run.py -gpu GPU_DEVICE --mode 1 --patient PATIENT`

## Using Docker
### requirements
Tested with Docker version 19.03.6, build 369ce74a3c on Ubuntu 18.04

### Build Image
Build Docker Image by running:

`docker build --tag nv1x16_eval .`

### Execution
Specify (any) root directory of your data segments by

`export DATA_DIR=YOUR_DATA_DIRECTORY`,

replacing `YOUR_DATA_DIRECTORY` with your specific directory.

Run training by executing

`docker run --gpus 1 -v $PWD:/code -v /$DATA_DIR:/$DATA_DIR:ro nv1x16_eval python ./run.py -gpu 0 --mode 1 -p PATIENT`

## options and remarks
For detailed options, run:

`python run.py -h`

The .csv files containing the paths of the segments (and targets for training) are expected to be in 
the current working directory if not specified by the argument

`--path PATH_TO_CSV`

For approximate reproducing the results from the paper (not exactly possible due to different 
training process) execute:

`python run.py -gpu GPU_DEVICE --mode 1 -p PATIENT`

to train the model. Warning: Displayed metrics that cannot be calculated accumulatively (roc_auc, 
precision, recall, pr_auc) are only batch-wise approximations. 

Depending on Hardware, you may want or have to change batchsize by specifying argument

`--batch_size BATCH_SIZE`

Then you can validate it with

`python run.py --mode 2 --patient PATIENT`

Testing works accordingliy.

Validation/Testing can also be done on gpu if you specify a device. I did not yet constrain the
implementation to run only single-threaded or measure maximum ram used during evaluation. 

Length of the file segments can be specified with option `-l 1` or `l 10 (default)`, subtract mean can 
switched on by adding `-sm (default off)`. Should not make any difference though, since data is standardized
anyways in training procedure.

All models, training logs and settings are saved in the `archive` directory. If you do not specify any 
model when testing using `--model_file MODEL_FILE`, last model that was trained with the given evaluation 
settings (segment length, subtract_mean, patient) will be evaluated.

Since I could not test it in the original setup with the original csv files and data, it is possible
that there are bugs that I did not think of at this point.

Do not hesitate to contact me if you have any questions or remarks.



