import torch as T
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch
from torch.autograd import Variable

# Hyper Parameters

class Actor(nn.Module):
    def __init__(self, space_states_dim, space_actions_dim, device, LAYER1_SIZE= 300, LAYER2_SIZE= 200):
        super(Actor, self).__init__()
        self.device = device
        self.fc1= nn.Linear(space_states_dim, LAYER1_SIZE).to(self.device)

        self.fc2= nn.Linear(LAYER1_SIZE, LAYER2_SIZE).to(self.device)

        self.fc3= nn.Linear(LAYER2_SIZE, space_actions_dim).to(self.device)


    def forward(self, state):

        l1 = F.relu(self.fc1(state)).to(self.device)
        l2 = F.relu(self.fc2(l1)).to(self.device)
        actions = T.tanh(self.fc3(l2))

        return actions

class Critic(nn.Module):
    def __init__(self, space_states_dim, space_actions_dim, device, LAYER1_SIZE= 300, LAYER2_SIZE= 200):
        super(Critic, self).__init__()
        self.device = device
        self.fc1= nn.Linear(space_states_dim+space_actions_dim, LAYER1_SIZE).to(self.device)
        self.fc2= nn.Linear(LAYER1_SIZE, LAYER2_SIZE).to(self.device)
        self.fc3= nn.Linear(LAYER2_SIZE, 1).to(self.device)


    def forward(self, state, action):
        x =T.cat([state, action],1).to(self.device) # ----- 4 -----
        l1 = F.relu(self.fc1(x)).to(self.device)
        l2 = F.relu(self.fc2(l1)).to(self.device)
        Q = self.fc3(l2)

        return Q
    

class ActorLSTM(nn.Module):
    def __init__(self, space_states_dim, space_actions_dim, device, LAYER1_SIZE= 300, LAYER2_SIZE= 200):
        super(ActorLSTM, self).__init__()
        self.device = device
        self.space_states_dim = space_states_dim

        self.hidden_tensor = torch.FloatTensor(2, 2, LAYER1_SIZE).to(device=self.device)
        self.cell_tensor = torch.FloatTensor(2, 2, LAYER1_SIZE).to(device=self.device)

        self.hidden_shape = (self.hidden_tensor, self.cell_tensor)

        self.lstm = nn.LSTM(space_states_dim, LAYER1_SIZE, 1, batch_first=True).to(device=self.device)
        self.fc2= nn.Linear(LAYER1_SIZE, LAYER2_SIZE).to(self.device)
        self.fc3= nn.Linear(LAYER2_SIZE, space_actions_dim).to(self.device)


    def forward(self, state):
        x, _ = self.lstm(state.reshape((1,1,self.space_states_dim)))
        l2 = F.relu(self.fc2(x)).to(self.device)
        actions = T.tanh(self.fc3(l2))

        return actions

class CriticLSTM(nn.Module):
    def __init__(self, space_states_dim, space_actions_dim, device, LAYER1_SIZE= 300, LAYER2_SIZE= 200):
        super(CriticLSTM, self).__init__()
        self.device = device
        self.space_states_dim = space_states_dim+space_actions_dim

        self.hidden_tensor = torch.FloatTensor(1, 1, LAYER1_SIZE).to(device=self.device)
        self.cell_tensor = torch.FloatTensor(1, 1, LAYER1_SIZE).to(device=self.device)

        self.hidden_shape = (self.hidden_tensor, self.cell_tensor)

        self.lstm = nn.LSTM(self.space_states_dim, LAYER1_SIZE, 1, batch_first=True).to(device=self.device)
        self.fc2= nn.Linear(LAYER1_SIZE, LAYER2_SIZE).to(self.device)
        self.fc3= nn.Linear(LAYER2_SIZE, 1).to(self.device)


    def forward(self, state, action):
        x =T.cat([state, action],1).to(self.device) # ----- 4 -----
        lstm, _ = self.lstm(x.reshape((1,1,self.space_states_dim)))
        l2 = F.relu(self.fc2(lstm)).to(self.device)
        Q = self.fc3(l2)

        return Q