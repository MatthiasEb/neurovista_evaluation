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

You should use a GPU for training. I used an RTX 2080 Ti, run_on_contest_data=1, mode=1 took about 4.5 h.
If you use a GPU with much less RAM, you might have to reduce the batch size, I did not try that though.
I ran the code with run_on_contest_data=1, the results seemed to be comparable to the version on the ecosystem leaderboard. 
Sparse tests with run_on_contest_data=0 have been executed, maybe there is something I missed here. 
I did not yet try to run it within a singularity container, docker should work though.
Do not hesitate to contact me if you run into problems, have any questions or remarks.

### Algorithm
This is a pretty naive approach on a 2D-Convolution Deep Neural Network, applied to the raw time series. 
As described in the [paper](https://ieeexplore.ieee.org/abstract/document/8621225), the Network expects standardized 15 s segments, sampeled at 200 Hz. 
tensorflow.keras (2.0.1) was used as Deep Learning API. 
In order to avoid the need to either load the whole training set at once or to save the preprocessed time series, this is a different implementation than the one used in the paper.
In order to allow training on arbitrary large datasets, this implementation does not perfectly reproduce the results shown in the paper. 
I did a few testruns on the contest data, ROC AUC of the private Set should be around .25, .7 and .8 for Patient 1, 2 and 3 respectively. 
However, considerable variations are conceivable, see section below.

### Implementation
Loading the original (~ 400 Hz) .mat files, resampling to 200 Hz, standardizing (optionally, if `subtract_mean==1`), splitting them in 15 s segments is done asynchronously on the fly by the dataloader in 5 different threads.
The 15s Segments are enqueued in a buffer with the size of 400 10-min-sequences, implemented as a tf.queue.RandomShuffleQueue.
The data is therefore dequeued in random order, although not perfectly uniformly shuffeled, depending on the buffer size and the size of the data set. 
I did some experiments that showed that the buffersize can have considerable impact on the performance of the algorithm.
The bigger the buffer size, the closer are the results to the ones shown in the paper.
The intention of this procedure was to ensure a reasonably shuffeled training set of 15 s segments while minimizing IO, working on the .mat files and having the possibility for standardization.  
If the IO-Bandwidth of the filesystem is reasonably high, this should not slow down the training too much. 

As described in the paper, if run_on_contest_data==1, 3 networks (one for each patient) are trained and evaluated individually. 
Subsequently, the solution file is concatenated.





