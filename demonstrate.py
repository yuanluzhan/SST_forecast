
from ast import arg
from pyexpat import features
import time
import sys
import datetime
import os
import argparse
from unittest import result
import torch
import random
import pandas as pd
import numpy as np
from data.DataPreprocessing import build_s_a
from algorithms.dqn import dqn_LSTM
import json
import matplotlib.pyplot as plt

# from predictor.LSTMpredictor import LSTM

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sklearn.cluster import KMeans
from sklearn import metrics


def dqntrain(length, time_delay, episodes, dqn_lr, batch_dqn, reward, model_dir, action_space, batch_lstm, state_space,
             explore):
    # 数据导入与预处理
    # dqn = torch.load("results/dqn/2022_01_26_21_59_57/50dqn.pth")
    # for i in range(20):
    #     print(torch.tensor([i]))
    #     action = dqn.choose_action(torch.tensor([float(i)]).cuda(),0)
    #     print(action)
    # raise Exception('s')
    import time
    data = pd.read_csv("data/sst_1.csv", encoding='UTF-8')
    data = data.iloc[:, 0]
    max = data[data.argmax()]
    min = data[data.argmin()]
    mean = np.mean(data)
    for i in range(len(data)):
        data[i] = data[i] - mean

    states, labels = build_s_a(data, int(length * time_delay), 1)
    km = KMeans(n_clusters=state_space)
    y_pred = km.fit_predict(states)

    features_lstm, label_lstm = build_s_a(data, length, 1)
    features_lstm = torch.tensor(features_lstm).to(torch.float32).cuda()
    labels_lstm = torch.tensor(label_lstm).to(torch.float32).cuda()

    # states 预处理一下

    states = torch.tensor(states).to(torch.float32).cuda()
    labels = torch.tensor(labels).to(torch.float32).cuda()
    rl_states = torch.tensor(y_pred).to(torch.float32).cuda()
    rl_states = rl_states.reshape(len(rl_states), 1)
    if length ==2:
        predict_net = torch.load("results/LSTM/2022_02_05_17_09_18/predict250.pth")
    if length == 5:
        predict_net = torch.load("results/LSTM/2022_02_05_16_57_20/predict950.pth")
    if length == 10:
        predict_net = torch.load("results/LSTM/2022_02_05_16_56_32/predict950.pth")
    if length == 20:
        predict_net = torch.load("results/LSTM/2022_01_25_18_36_19/predict150.pth")
    if length == 30:
        predict_net = torch.load("results/LSTM/2022_02_05_16_57_02/predict950.pth")
    if length == 40:
        predict_net = torch.load("results/LSTM/2022_02_07_18_20_00/predict950.pth")
    if length == 50:
        predict_net = torch.load("results/LSTM/2022_02_07_18_20_37/predict950.pth")
    if length == 60:
        predict_net = torch.load("results/LSTM/2022_02_07_18_21_00/predict950.pth")
    print(predict_net)
    result_rl = []
    result_lstm = []
    dqn_losses = []

    for episode in range(episodes):
        errors_lstm = []
        errors_rl = []
        dimensions = []
        improvements = []
        bests = []
        actions = []
        result_rl = []

        for step in range(states.shape[0] - 3):
            if step%10 ==0:
                print(step)
            state = states[step]
            label = labels[step]
            rl_state = rl_states[step]
            feature1 = state[-length:]
            feature1 = feature1.reshape(length, 1, 1)
            loss_lstm, prediction1 = predict_net.test_rl(feature1, label)
            prediction_lstm = prediction1[-1]
            loss1 = abs(prediction_lstm - label)
            # print(rl_state)
            delay = 1
            # 测试一下
            record = []
            loss_rl = []
            for dimension in range(length):
                # feature2 = state[state.shape[0] % delay:state.shape[0] + 1 - delay:delay]
                feature2 = state
                feature2 = feature2[-(length-dimension):]
                padding = torch.zeros(dimension).cuda()

                feature2 = torch.cat((feature2,padding),dim=0)
                feature2 = feature2.reshape(length,1,1)

                tmp,predicton2 = predict_net.test_rl(feature2,label)

                predict_rl = predicton2[length-1-dimension]
                loss2 = abs(predict_rl-label)
                errors_rl.append(loss2.item())
                record.append(loss1.item()-loss2.item())
                loss_rl.append(loss2.item())

            action = np.argmax(record)
            improvement = record
            actions.append(action)
            improvements.append(improvement)
            bests.append(record[action])
            result_rl.append(loss_rl)




            # 根据action选取数据


            # if step ==12:
            #     raise Exception('s')

        # 数据保存



        if episode  == 0:
            result = pd.DataFrame(data=actions)
            result.to_csv(model_dir + "/actions.csv")
            result = pd.DataFrame(data=improvements)
            result.to_csv(model_dir + "/improvements.csv")
            result = pd.DataFrame(data=bests)
            result.to_csv(model_dir + "/bests.csv")
            result = pd.DataFrame(data=result_rl)
            result.to_csv(model_dir + "/result_rl.csv")
            # 状态表示y_pred =
            # plt.figure(figsize=(30, 30))
            # plt.subplot(3, 1, 1)
            # time = np.arange(0, len(errors_lstm))
            # plt.scatter(time, y_pred[:len(errors_lstm)], label="state")
            # # plt.plot_date(time,errors_rl,linestyle='-',markersize=0.1,label="rl")
            # plt.xticks(time[0::60], rotation="vertical", fontsize="xx-large")
            # plt.yticks
            # plt.legend()
            # plt.subplot(3, 1, 2)
            # plt.scatter(time, dimensions, label="dimensions")
            # # plt.plot_date(time,errors_rl,linestyle='-',markersize=0.1,label="rl")
            # plt.xticks(time[0::60], rotation="vertical", fontsize="xx-large")
            # plt.yticks
            # plt.legend()
            # plt.subplot(3, 1, 3)
            # plt.scatter(time, improvements, label="improvements")
            # # plt.plot_date(time,errors_rl,linestyle='-',markersize=0.1,label="rl")
            # plt.xticks(time[0::60], rotation="vertical", fontsize="xx-large")
            # plt.yticks
            # plt.legend()
            # plt.savefig(model_dir + "/result" + str(episode) + ".png")
            # print("plotlstm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--length', help="the length of the sequence", type=int, default=40)
    parser.add_argument('--time_delay', help="the length of the sequence", type=int, default=1)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--dqn_lr', type=float, default=0.0001)
    parser.add_argument('--batch_dqn', type=int, default=10)
    parser.add_argument('--action_space', type=int, default=20)
    parser.add_argument('--state_space', type=int, default=7)
    parser.add_argument('--reward', type=float, default=0.1)
    parser.add_argument('--model_dir', default="results/demonstrate/" + time.strftime('%Y_%m_%d_%H_%M_%S'))
    parser.add_argument('--explore', type=float, default=0.05)

    args = parser.parse_args()
    os.makedirs(args.model_dir)
    with open(args.model_dir + "/params.json", mode="w") as f:
        json.dump(args.__dict__, f)

    dqntrain(args.length, args.time_delay, args.episodes, args.dqn_lr, args.batch_dqn, args.reward, args.model_dir,
             args.action_space, 10, args.state_space, args.explore)
