# eval_repo

This repository contains the evaluation scripts for evaluating the metrics discussed in the final report. The three main categories of evaluation are Metric Accuracy, Semantic Accuracy and Practicality.
---

## Project Structure
```text
EVAL_REPO/
├── code/                     
│   ├── get_gt3.py               #gets ground-truth
│   ├── main.py                  #gets Hydra and Clio eval results
│   ├── s-graph_eval.py          #gets S-Graphs+ eval results
│   ├── stats_functions.py       #helper script for doing the calculations
│   └── timings.py               #gets timings stats for all methods
├── graphs/                      
│   ├── clio/
│   │   └── eval_runs_20260328_1754/
│   │       ├── run_1/           # run data
│   │       ├── run_2/           # run data
│   │       ├── run_3/           # run data
│   │       ├── run_4/           # run data
│   │       └── run_5/           # run data
│   ├── hydra/
│   │   └── eval_runs/
│   │       ├── run_1/           # run data
│   │       └── ...
│   └── s-graphs/
│       ├── run_1/               # run data
│       └── ...
├── media/                       
├── final.world                  #gazebo world for ground truth
├── ground_truth_objects.csv     #ground truth object data
├── README.md
└── requirements.txt
```

# SETUP GUIDE:

## Create Python Virtual Environment:

### Create and activate the venv
```
#create venv
python3 -m venv dsg_eval_env

#activate venv
source dsg_eval_env/bin/activate
```
### Install pip dependencies
```
pip install -r requirements.txt
```

# USAGE GUIDE:

This section explains how to run each script which was used to generate the results in the final report.
---

### Navigate to code directory
```
(activate venv)

cd eval_repo/code
```
### Print Hydra and Clio Data:
```
python3 main.py
```
Outputs: dataframes for evaluation metrics

### Print S-Graphs+ Data:
```
python3 s-graph_eval.py
```
Outputs: dataframes for evaluation metrics

### Print Timings Data:
```
python3 timings.py
```
Outputs: dataframes for timings metrics
