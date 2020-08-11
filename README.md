# neurovista_evaluation
code for evaluation on full neurovista trial data following the [instructions](https://github.com/epilepsyecosystem/CodeEvaluationDocs) (commit 20e6f0f, dated 16/06/2020). 

## Settings
Settings can be adjusted in SETTINGS.json

## Using a virtual environment
### requirements
1. python3
2. cuda toolbox 10, nvidia-drivers
 
### installation
Install requirements by running:

`pip install -r requirements.txt`


### execution
Run training by executing:

`python run.py`

## Using Docker
### requirements
Tested with Docker version 19.03.6, build 369ce74a3c on Ubuntu 18.04

### Build Image
Build Docker Image by running:

`docker build --tag nv1x16_eval .`

### Execution
Specify the directory of your data segments by

`export DATA_DIR=YOUR_DATA_DIRECTORY`,

replacing `YOUR_DATA_DIRECTORY` with your specific directory.

Run training by executing

`docker run --gpus 1 -v $PWD:/code -v /$DATA_DIR:/$DATA_DIR:ro nv1x16_eval python ./run.py`

## Using Singularity

Singularity recipe is included. SingularityHub URI of the Image is MatthiasEb/neurovista_evaluation:nv_eval.

## Remarks

I ran with run_on_contest_data=1, the results seemed to be comparable to the version on the ecosystem leaderboard. 
Sparse tests with run_on_contest_data=0 have been executed, maybe there is something I missed here. 
I did not yet try to run it within a singularity container, docker should work though.
Do not hesitate to contact me if you run into problems, have any questions or remarks.

### Algorithm
This is a pretty naive approach on a 2D-Convolution Deep Neural Network on the raw time series. 
As described in the [paper](https://ieeexplore.ieee.org/abstract/document/8621225), the Network expects standardized 15 s segments, sampeled at 200 Hz. 
tensorflow.keras (2.0.1) was used as Deep Learning API. 
In order to avoid the need to either load the whole training set at once or to save the preprocessed time series, this is a slightly different implementation than the one used in the paper. 
Loading the original .mat files, standardizing (optionally, if `subtract_mean==1`), splitting them in 15 s segments and subsequently shuffling the data is (hopefully) done asynchronously on the fly by the dataloader. 
If the IO-Bandwidth of the filesystem is large enough, this should not pose the bottleneck during training. 
Subsampling is implemented as an AveragePooling Input Layer in the network. 

As described in the paper, if run_on_contest_data, 3 networks (one for each patient) is trained and evaluated separately. 
Subsequently, the solution file is concatenated.





