# Readme

code for using DRL in SST forecast

### Installation

Requirements:

```
torch
numpy
```

### Structure

```
|-- SST_forecast
    |-- algorithms # RL algorthms
    |-- data # train data
    |-- plot #data for plot
    |-- predictors # predictors like LSTM, NN
    |-- README.md
    ......
```

### Usage

Predict the SST

```shell
python mainLSTM.py 
```

Employing RL in SST_predict, i.e. , using RL to select the data.

```
python maindqn.py 
```

