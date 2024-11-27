import os
import random
import torch.nn as nn
import torch

class DeepQNetwork(nn.Module):
  def __init__(self, n_actions=8, input_dim=680, hidden_dim=64):
    super(DeepQNetwork, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, n_actions)

    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

    self._create_weights()

  def _create_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = x.reshape(x.size(0), -1)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)

    return x


class Agent:
  def __init__(self, turn, epsilon):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    weight_file_path = os.path.join(dir_path, turn, 'weight')
    self.network = DeepQNetwork()
    self.network.load_state_dict(torch.load(weight_file_path))

  def choose_action(self, observation):
    state = torch.flatten(torch.tensor(observation[:, :17])).to(self.network.device)
    _, advantage = self.network.forward(state)
    action = torch.argmax(advantage).item()
    return action
