from ast import arg
from pyexpat import features
import time
import sys
import datetime
import os
import argparse
from unittest import result

import numpy
import torch
import random
import pandas as pd
import numpy as np
from data.DataPreprocessing import build_s_a
from algorithms.dqn import dqn_LSTM
import json
import matplotlib.pyplot as plt

def vision_dqn(length,n_features,lstm_batch,model_dir,n_days):
    data1 = pd.read_csv("data/Nan_Hai.csv")
    data1 = data1[-400:1]


    #dqn = torch.load("result_bohai/dqn/2022_04_02_15_07_32/850dqn.pth")
    #predict_net = torch.load("result_bohai/LSTM3/2022_03_31_11_40_30/predict.pth")
    dqn = torch.load("result_nanhai/dqn/2022_04_05_21_15_45/800dqn.pth")
    predict_net = torch.load("result_nanhai/dqn/2022_04_05_21_15_45/lstm.pth")
    result_lstm = []
    result_rl = []
    result_dim = []
    result_ground = []
    for j in range(data1.shape[1]-10):

        errors_lstm = []
        errors_rl = []
        dimensions = []
        predictions_lstm = []
        groundtruths = []
        data = data1.iloc[:, j+1:j+2].values
        mean = np.mean(data)
        for i in range(len(data)):
            data[i] = data[i] - mean
        states, label = build_s_a(data, length, n_days)
        states = torch.tensor(states).to(torch.float32).cuda()
        labels = torch.tensor(label).to(torch.float32).cuda()
        for step in range(states.shape[0] - 3):
            state = states[step]
            label = labels[step]
            feature1 = state[-length:, :]
            feature1 = feature1.reshape(length, 1, n_features)
            loss_lstm, prediction1 = predict_net.test_rl(feature1, label,1,1)
            prediction_lstm = prediction1[-1]
            loss1 = abs(prediction_lstm - label[-1])
            predictions_lstm.append(prediction_lstm.item())
            errors_lstm.append(torch.mean(loss1).item())
            groundtruths.append(label[-1].item())


            #DQN 选取动作
            state1 = state.reshape(state.shape[0] * state.shape[1])
            action = dqn.choose_action(state1, 0)


            dimension = int(action.item())  # * (length//action_space))
            dimensions.append(length - dimension)
            feature2 = state
            feature2 = feature2[-(length - dimension):]
            padding = torch.zeros(dimension, n_features).cuda()
            feature2 = torch.cat((feature2, padding), dim=0)
            feature2 = feature2.reshape(length, 1, n_features)
            tmp, predicton2 = predict_net.test_rl(feature2, label,1,1)
            predict_rl = predicton2[length - 1 - dimension]
            loss2 = abs(predict_rl - label[-1]).detach()
            errors_rl.append(torch.mean(loss2).item())
        result_lstm.append(errors_lstm)
        result_rl.append(errors_rl)
        result_dim.append(dimensions)
        result_ground.append(groundtruths)
    print("ylz")
    result = pd.DataFrame(data=result_ground)
    result.to_csv(model_dir + "/groundtruth.csv")
    result = pd.DataFrame(data=result_rl)
    result.to_csv(model_dir + "/errors_rl.csv")
    result = pd.DataFrame(data=result_lstm)
    result.to_csv(model_dir + "/errors_lstm.csv")
    result = pd.DataFrame(data=result_dim)
    result.to_csv(model_dir + "/dimensions.csv")
    time = np.arange(0, len(errors_lstm))
    plt.scatter(time, errors_rl, label="improvements")
    plt.show()

def plot_dqn(model_dir):
    errors_rl=pd.read_csv(model_dir + "/errors_rl.csv").values.transpose()
    errors_lstm = pd.read_csv(model_dir + "/errors_lstm.csv").values.transpose()
    groundtruths = pd.read_csv(model_dir + "/groundtruth.csv").values.transpose()
    dimensions = pd.read_csv(model_dir + "/dimensions.csv").values.transpose()
    improvements = []
    time_diff = []
    imd_diff = []
    for i in range(errors_rl.shape[0]):
        # print(errors_lstm[i,1])
        # print(errors_rl[i,1])
        improvements.append(errors_lstm[i,1]-errors_rl[i,1])
        if errors_lstm[i,1]-errors_rl[i,1]!=0:
            time_diff.append(i)
            imd_diff.append(errors_lstm[i,1]-errors_rl[i,1])
    print(imd_diff)
    plt.figure(figsize=(30, 30))
    plt.title("10days")
    plt.subplot(4, 1, 1)
    time = np.arange(0, len(errors_lstm))
    plt.plot(time, groundtruths[:,1], label="groundtruth")
    # plt.plot_date(time,errors_rl,linestyle='-',markersize=0.1,label="rl")
    plt.xticks(time[0::60], rotation="vertical", fontsize="xx-large")
    plt.yticks
    plt.legend(fontsize=30)
    plt.subplot(4, 1, 2)
    plt.hist(dimensions[:,1], bins=30,range=(0,30),label="dimensions")
    # plt.scatter(time, dimensions[:,1], label="dimensions")
    # plt.xticks(time[0::60], rotation="vertical", fontsize="xx-large")
    plt.yticks
    plt.legend(fontsize=30)
    plt.subplot(4, 1, 3)
    plt.scatter(time_diff, imd_diff, label="groundtruth")
    # plt.plot_date(time,errors_rl,linestyle='-',markersize=0.1,label="rl")
    plt.xticks(time[0::60], rotation="vertical", fontsize="xx-large")
    plt.yticks
    plt.legend(fontsize=30)
    plt.subplot(4, 1, 4)
    plt.hist(imd_diff,bins=100,range=(-0.001,0.001),label="improvements")
    # plt.plot_date(time,errors_rl,linestyle='-',markersize=0.1,label="rl")
    # plt.xticks(time[0::60], rotation="vertical", fontsize="xx-large")
    # plt.yticks
    plt.legend(fontsize=30)


    plt.savefig("10days.png")




def plot_on_map():
    data = pd.read_csv("groundtruth.csv").values
    print(data.shape)
    print(data[:,0:2])
    print(data[:, 2].shape)
    map = folium.Map(location=[10,115],zoom_start=10)
    HeatMap(data).add_to(map)

    map.save("a.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--length', help="the length of the sequence",type=int,default=30)
    parser.add_argument('--time_delay', help="the length of the sequence",type=int,default=1)
    parser.add_argument('--episodes',type=int , default=10000)
    parser.add_argument('--dqn_lr',type=float, default=0.0001)
    parser.add_argument('--dqn_batch',type=int,default=10)
    parser.add_argument('--lstm_batch', type=int, default=10)
    parser.add_argument('--n_features', type=int, default=1)
    parser.add_argument('--reward',type=float,default=0.1)
    parser.add_argument('--model_dir',default="plot")
    parser.add_argument('--n_days',type=int,default=3)
    
    args = parser.parse_args()
    # os.makedirs(args.model_dir)
    # plot1()
    vision_dqn(args.length, args.n_features, args.lstm_batch,args.model_dir,args.n_days)
    # vision_lstm(args.length,args.n_features,args.lstm_batch)
    plot_dqn(args.model_dir)
    # vision(args.length, args.time_delay, args.model_dir)