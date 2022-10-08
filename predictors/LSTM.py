#实现一个不同窗口的predictor
#输入为不同窗口大小的数据，输出为预测值
import numpy as np
import torch
import torch.nn as nn

class predictorLSTM(nn.Module):
    def __init__(self,n_features, hidden_layer_size, batch_size, lstm_lr):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(n_features, hidden_layer_size,num_layers=1).cuda()
        

        self.hidden_cell = (torch.zeros(n_features,batch_size,self.hidden_layer_size).cuda(),
                            torch.zeros(n_features,batch_size,self.hidden_layer_size).cuda())
        
        self.linear =  nn.Sequential(
            nn.Linear(hidden_layer_size,hidden_layer_size).cuda(),
            nn.ReLU().cuda(),
            nn.Linear(hidden_layer_size, hidden_layer_size).cuda(),
            nn.ReLU().cuda(),
            nn.Linear(hidden_layer_size,n_features).cuda()
        )
        

        self.loss = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lstm_lr,)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        result = self.linear(lstm_out)
        return result
    
    def train(self, input_seq, test_seq):
        self.optimizer.zero_grad()
        self.hidden_cell = (torch.zeros(1,self.batch_size,self.hidden_layer_size).cuda(),
                            torch.zeros(1,self.batch_size,self.hidden_layer_size).cuda())
        prediction = self(input_seq)
        # prediction 是一个tensor(L,N,1),loss真实的物理含义是什么。
        loss1 = self.loss(prediction, test_seq)
        loss2 = self.loss(prediction[-1,:,:],test_seq[-1,:,:])
        loss = loss1

        
        loss.backward()
        self.optimizer.step()
        return prediction,loss

    def test(self, input_seq,test_seq):
        
        # 加一个不计算 torch_no_grad
        self.hidden_cell = (torch.zeros(1,self.batch_size,self.hidden_layer_size).cuda(),
                            torch.zeros(1,self.batch_size,self.hidden_layer_size).cuda())
        with torch.no_grad():
            prediction = self(input_seq)


        loss = self.loss(prediction, test_seq)

        return prediction,loss

    
    def test_rl(self, input_seq, test_labels,dqn_batch_size,n_features):
        self.hidden_cell = (torch.zeros(1,dqn_batch_size,self.hidden_layer_size).cuda(),
                            torch.zeros(1,dqn_batch_size,self.hidden_layer_size).cuda())
        
        with torch.no_grad():
            prediction = self(input_seq)

        loss = self.loss(prediction[-1,:,:], test_labels)

        return loss,prediction


