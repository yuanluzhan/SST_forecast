
from pickletools import optimize
import time
import sys
import datetime
import os
import argparse
from cv2 import split
import torch
import torch.nn as nn
import random
import pandas as pd
import numpy as np
from data.DataPreprocessing import build_s_a
from predictors.LSTM import predictorLSTM
import json
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
SPLIT_RATE=0.75


def lstmtrain(length,n_features,epochs,lstm_lr,lstm_batch,hidden_size,model_dir,n_days):
    # 数据导入与预处理
    data3 = pd.read_csv("data/Bo_Hai.csv")
    data = data3.iloc[:,1:n_features+1].values
    mean = np.mean(data)
    for i in range(len(data)):
        data[i] = data[i] - mean
    features, label= build_s_a(data, length, n_days)

    np.random.seed(5)
    np.random.shuffle(features)
    np.random.seed(5)
    np.random.shuffle(label)


    
    split_index = int(len(label)*SPLIT_RATE)
    test_index = int(len(label)-split_index)
    features = torch.tensor(features).to(torch.float32).cuda()
    labels = torch.tensor(label).to(torch.float32).cuda()

    
    # 训练集合测试集的分布不一致，用现在这种的训练集和测试集的划分方式。
    train_features, train_labels= features[:split_index], labels[:split_index]

    test_features, test_labels = features[split_index:], labels[split_index:]


    predict_net=predictorLSTM(n_features,hidden_size,lstm_batch,lstm_lr)
    print(predict_net)
    train_maes=[]
    test_maes=[]

    for epoch in range(epochs):
        train_losses = []
        # 训练的部分
        for step in range(int(split_index//lstm_batch)-2):
            index = range(int(lstm_batch*step),int(lstm_batch*(step+1)))
            input_seq = train_features[index]
            out_groundtruth = train_labels[index]
            # print(input_seq)

            input_seq = input_seq.reshape(length,lstm_batch,n_features)
            out_groundtruth =out_groundtruth.reshape(length,lstm_batch,n_features)
            
            prediction, loss = predict_net.train(input_seq,out_groundtruth)
            train_losses.append(loss.item())
        
        print('Pretrain Episode %d, MAE: %.4f' % (epoch,np.mean(train_losses)))

        #测试部分
        test_losses = []
        test_losses_last = []
        for step in range(int(test_index//(lstm_batch))-2):
            index = range(int(lstm_batch*step),int(lstm_batch*(step+1)))
            input_seq = test_features[index]
            test_seq= test_labels[index]

            input_seq = input_seq.reshape(length,lstm_batch,n_features)
            test_seq = test_seq.reshape(length,lstm_batch,n_features)
            prediction,test_loss = predict_net.test(input_seq,test_seq)
            test_loss_last = abs(prediction[-1,:,:]- test_seq[-1,:,:])
            test_losses.append(torch.mean(test_loss).item())
            test_losses_last.append(torch.mean(test_loss_last).item())
        print('Episode %d MAE_test : %.4f,last:%.4f' % (epoch, np.mean(test_losses),np.mean(test_losses_last)))
        train_maes.append(np.mean(train_losses))
        test_maes.append(np.mean(test_losses))


        # 模型保存 还有优化的空间

        if epoch%50==0:
            torch.save(predict_net, model_dir + "/predict.pth")
            results=pd.DataFrame(data=train_maes)
            results.to_csv(model_dir+"/train_mae.csv")
            results=pd.DataFrame(data=test_maes)
            results.to_csv(model_dir+"/test_mae.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--length', help="the length of the sequence",type=int,default=30)
    parser.add_argument('--time_delay', help="the length of the sequence",type=int,default=1)
    parser.add_argument('--epoch',type=int , default=300)
    parser.add_argument('--lstm_lr',type=float, default=0.0001)
    parser.add_argument('--lstm_batch',type=int,default=10)
    parser.add_argument('--lstm_hidden_size',type=int,default=500)
    parser.add_argument('--model_dir',default="result_Bohai/LSTM3/"+ time.strftime('%Y_%m_%d_%H_%M_%S'))
    parser.add_argument('--n_features', type=int,default=1)
    parser.add_argument('--n_days', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.model_dir)
    with open(args.model_dir+"/0params.json", mode="w") as f:
        json.dump(args.__dict__, f)
    

    lstmtrain(args.length,args.n_features, args.epoch, args.lstm_lr, args.lstm_batch,args.lstm_hidden_size,args.model_dir,args.n_days)