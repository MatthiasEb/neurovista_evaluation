# neurovista_evaluation
code for evaluation on full neurovista trial data

## requirements
1. python3
2. cuda toolbox 10.0, nvidia-drivers

## installation
`pip install -r requirements.txt`

## execution
For detailed options, run:

`python run.py -h`

For approximate reproducing the results from the paper (not exactly possible due to different 
training process) execute:

`python run.py --gpu_device 0 --mode 1 --patient 1 --subtract_mean 1 `