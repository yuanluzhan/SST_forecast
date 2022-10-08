
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
import math

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))
def dqntrain(length,n_features,episodes,dqn_lr,batch_dqn,dqn_hidden_size,model_dir,n_days):

    data = pd.read_csv("data/Bo_Hai.csv")
    data = data.iloc[10000:12000, 1:n_features+1].values
    mean = np.mean(data)
    for i in range(len(data)):
        data[i] = data[i] - mean

    states, labels= build_s_a(data, length, n_days)
    states = torch.tensor(states).to(torch.float32).cuda()
    labels = torch.tensor(labels).to(torch.float32).cuda()


    dqn = dqn_LSTM(length*n_features,length ,batch_dqn, dqn_hidden_size)

    if length == 3:
        predict_net = torch.load("result/LSTM/2022_02_28_19_53_47/predict950.pth")
    if length ==14:
        predict_net = torch.load("result/LSTM/2022_02_28_19_53_15/predict950.pth")
    if length == 5:
        predict_net = torch.load("results/LSTM/2022_02_05_16_57_20/predict950.pth")
    if length == 7:
        predict_net = torch.load("result/LSTM/2022_02_28_19_52_47/predict950.pth")
    if length == 20:
        predict_net = torch.load("results/LSTM/2022_01_25_18_36_19/predict150.pth")
    if length == 30:
        # dong hai
        # predict_net = torch.load("result_donghai/LSTM3/2022_03_11_10_38_22/predict950.pth")
        #bo hai
         
        #if n_days==7:
        #    predict_net = torch.load("result_Bohai/LSTM3/2022_04_07_14_04_18/predict.pth")
        if n_days==3:
            predict_net = torch.load("result_Bohai/LSTM3/2022_04_07_16_37_18/predict.pth")
        # predict_net = torch.load("result_bohai/LSTM3/2022_03_31_11_40_30/predict.pth")
        # nan hai
        #n_days=7
        
        #if n_days==7:
        #    predict_net = torch.load("result_nanhai/LSTM3/2022_04_05_15_02_44/predict.pth")
        #if n_days==3:
        #    predict_net = torch.load("result_nanhai/LSTM3/2022_04_05_15_02_01/predict.pth")
        #SST_1
        # predict_net = torch.load("result/LSTM/2022_02_28_19_53_34/predict950.pth")
    if length == 40:
        predict_net = torch.load("result/LSTM/2022_02_22_23_04_40/predict50.pth")
    if length == 50:
        predict_net = torch.load("results/LSTM/2022_02_07_18_20_37/predict950.pth")
    if length == 60:
        predict_net = torch.load("result/LSTM/2022_02_20_23_15_38/predict950.pth")

    result_rl = []
    result_lstm = []
    result_dqn = []
    for episode in range(episodes):
        errors_lstm = []
        errors_rl = []
        dqn_losses = []
        dimensions = []
        improvements = []
        Q_means = []
        reward_means = []
        for step in range(int(states.shape[0]-3)):
            state = states[step]
            label = labels[step]

            feature1 = state.reshape(length,1,n_features)
            loss_lstm,prediction1 = predict_net.test_rl(feature1,label,1,n_features)
            prediction_lstm = prediction1[-1]
            loss1 = abs(prediction_lstm-label[-1]).detach()
            errors_lstm.append(torch.mean(loss1).item())
            state1 = state.reshape(state.shape[0]*state.shape[1])
            if episode < 10:
                action = dqn.choose_action(state1, 1)
            elif episode%50==0:
                action = dqn.choose_action(state1, 0)
            else:
                action = dqn.choose_action(state1, 0.7)

            dimension = int(action.item())# * (length//action_space))
            dimensions.append(length-dimension)
            feature2 = state[-(length-dimension):]
            padding = torch.zeros(dimension,n_features).cuda()
            feature2 = torch.cat((feature2,padding),dim=0)
            feature2 = feature2.reshape(length,1,n_features)

            tmp,predicton2 = predict_net.test_rl(feature2,label,1,n_features)

            predict_rl = predicton2[length-1-dimension]
            loss2 = abs(predict_rl-label[-1]).detach()
            errors_rl.append(torch.mean(loss2).item())

            improvements.append(torch.mean(loss1-loss2).item())

            loss1_tmp = torch.mean(loss1)
            loss2_tmp = torch.mean(loss2)

            if loss1_tmp-loss2_tmp>0:
                reward = 10+1/(0.0001+loss2_tmp)
            elif loss1_tmp-loss2_tmp<0:
                reward = -10
            else:
                reward = 0

            next_state = states[step+1]
            dqn.store_transition(state1, action, reward, next_state)

            if episode<10:
                dqn_loss, Q_mean, reward_mean = dqn.learn(5*dqn_lr)
            else:
                dqn_loss, Q_mean, reward_mean = dqn.learn(dqn_lr)

            if dqn_loss!=0:
                dqn_losses.append(dqn_loss)
                Q_means.append(Q_mean.item())
                reward_means.append(reward_mean.item())

            if step % 10 ==0:
                dqn.target_net =dqn.current_net

        if episode%10==0:
            print("test")
            
        print('Episode %d MAE_RL: %.4f,MAE_LSTM:%.4f , DQN_loss %.4f' % (episode, np.mean(errors_rl), np.mean(errors_lstm), np.mean(dqn_losses)))
        print('Q_values:%.4f , reward_mean %.4f' % ( np.mean(Q_means), np.mean(reward_means)))

        result_rl.append(np.mean(errors_rl))
        result_lstm.append(np.mean(errors_lstm))
        result_dqn.append(np.mean(dqn_losses))

        if episode%50==0:
            torch.save(dqn, model_dir + "/" + str(episode) + "dqn.pth")
            torch.save(predict_net, model_dir + "/lstm.pth")
            result = pd.DataFrame(data=result_rl)
            result.to_csv(model_dir+"/RL_loss.csv")
            result = pd.DataFrame(data=result_lstm)
            result.to_csv(model_dir+"/LSTM_loss.csv")
            result = pd.DataFrame(data=result_dqn)
            result.to_csv(model_dir + "/DQN_loss.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--length', help="the length of the sequence",type=int,default=30)
    parser.add_argument('--time_delay', help="the length of the sequence",type=int,default=1)
    parser.add_argument('--episodes',type=int , default=10000)
    parser.add_argument('--dqn_lr',type=float, default=0.0001)
    parser.add_argument('--reward_type', type=str, default="+-100")
    parser.add_argument('--batch_dqn',type=int,default=10)
    parser.add_argument('--action_space',type=int,default=20)
    parser.add_argument('--dqn_hidden_size',type=int,default=5000)
    parser.add_argument('--reward',type=float,default=1)
    parser.add_argument('--model_dir',default="result_Bohai/dqn/"+ time.strftime('%Y_%m_%d_%H_%M_%S'))
    parser.add_argument('--explore',type=float,default=0.05)
    parser.add_argument('--n_features', type=int, default=1)
    parser.add_argument('--n_days', type=int, default=3)


    args = parser.parse_args()
    os.makedirs(args.model_dir)
    with open(args.model_dir+"/0params.json", mode="w") as f:
        json.dump(args.__dict__, f)
    


    dqntrain(args.length,args.n_features, args.episodes, args.dqn_lr, args.batch_dqn,args.dqn_hidden_size,args.model_dir,args.n_days)