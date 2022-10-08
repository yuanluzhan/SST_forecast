import random
from collections import namedtuple, deque

from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Qnet(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Qnet, self).__init__()
        self.Q_net=nn.Sequential(
            nn.Linear(n_input, n_hidden).cuda(),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden).cuda(),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Linear(n_hidden, n_hidden).cuda(),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_hidden, n_hidden).cuda(),

            nn.ReLU(),
            nn.Linear(n_hidden, n_output).cuda()

        )


    def forward(self, input):
        out = self.Q_net(input)
        return out


class dqn_LSTM(nn.Module):
    def __init__(self,inputs,outputs,batch_size,DQN_hidden_size):
        super(dqn_LSTM, self).__init__()

        self.current_net=Qnet(inputs,DQN_hidden_size,outputs).cuda()
        self.target_net = Qnet(inputs,DQN_hidden_size,outputs).cuda()
        self.target_net.load_state_dict(self.current_net.state_dict())
        #self.traget_net=Qnet(outputs)
        self.BATCH_SIZE=batch_size
        self.Gamma=0.01
        self.memory=ReplayMemory(1000)



    def choose_action(self,state,epsilon):
        #state=torch.tensor([state])
        
        state_action=self.current_net(state)
        state_action[0]=0
        #print(state_action)
        a=random.uniform(0,1)
        if a>epsilon:
            action = torch.argmax(state_action)
            # print("state_action")
            # print(state_action)
            #action = action.item()
        else:
            action= random.randint(0,state_action.shape[0]-2)
            action = torch.tensor(action)
        return action

    
    def store_transition(self,state,action,reward,next_state):
        action=torch.tensor([action]).to(torch.float32).cuda()
        reward=torch.tensor([reward]).to(torch.float32).cuda()
        #next_state=torch.tensor([next_state]).to(torch.float32).cuda()
        self.memory.push(state,action,next_state,reward)

    def learn(self,lr):
        #print(len(self.memory))
        if len(self.memory)<self.BATCH_SIZE+1:
            return 0,0,0
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(batch.state,dim=0)
        reward_batch = torch.stack(batch.reward,dim=0)
        next_state_batch=torch.stack(batch.next_state,dim=0)
        action_batch = torch.stack(batch.action,dim=0).to(torch.int64)
        
        state_action_values = self.current_net(state_batch).gather(1,action_batch)
        next_state_values = reward_batch
        expected_state_action_values=next_state_values
        #expected_state_action_values [0]=0

        
        
        criterion = nn.MSELoss()
        # print(state_action_values)
        # print(expected_state_action_values)
        # print(reward_batch)
        loss= criterion(state_action_values, expected_state_action_values)
        tmp = loss.item()
        # print(tmp)
        optimizer = optim.Adam(self.current_net.parameters(),lr=lr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(torch.mean(state_action_values))
        # print(torch.mean(reward_batch))
        # raise Exception('s')
        return tmp,torch.mean(state_action_values),torch.mean(reward_batch)
        




