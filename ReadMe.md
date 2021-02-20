# Perceive Your User in the Page Context: Recurrent Attention over Contextualized Page Sequence for Personalized E-commerce Search Ranking System

KDD 2021 Anonymous Submission #3322.


## Introduction
Pipeline:
1. Prepare Data
2. Train Model

## Running

We test our code on Python 3.7.5 and PyTorch 1.7.1.

### 1. Prepare Data

```
mkdir ./Data/raw/
mkdir ./Data/new/
mkdir ./Data/data/
cd ./Data/raw/
kaggle competitions download -c avito-context-ad-clicks
py7zr -x *.7z
cd ../../
python ./DataPreprocess/first_step.py
python ./DataPreprocess/second_step.py
```
When you see the files below, you can do the next work.
```
./Data/data/Train_data.csv
./Data/data/Val_data.csv
./Data/data/Test_data.csv
./Data/data/recent_history.pickle
```
### 2. Train Model
Firstly, set the `sys_path` in `./Runs/project_path.ini` as the absoluate path of current project.

Seconly, run the command `python ./Runs/run.py --model_name racp`.

