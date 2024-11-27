import random
import torch.nn as nn
import torch
import os

class DeepQNetwork(nn.Module):
  def __init__(self, input_dim=680, hidden_dim=64):
    super(DeepQNetwork, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, 1)

    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

    self._create_weights()

  def _create_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    # Flatten đầu vào để có thể vào các lớp Linear
    x = x.reshape(x.size(0), -1)
    # Chuyển (batch_size, 20, 17) thành (batch_size, 340)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)

    return x


class Agent:
  def __init__(self, model, epsilon=0.1):
    self.model = model
    self.epsilon = epsilon
    base_dir = "F:/Python/pythonProject/AI challenge/Double-Agent-Tetris/trained_models"
    weight_file_path = os.path.join(base_dir, "weight.pth")

    if os.path.exists(weight_file_path):
      self.model.load_state_dict(torch.load(weight_file_path, map_location=self.model.device))
      print(f"Loaded weights from {weight_file_path}")
    else:
      raise FileNotFoundError(
        f"Weights file not found at {weight_file_path}. "
        f"Please ensure the file exists and the path is correct.")

  def choose_action(self, state):
    if random.random() < self.epsilon:
      return random.randint(0, 7)  # Chọn ngẫu nhiên
    else:
      state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.model.device)
      q_values = self.model(state)
      return torch.argmax(q_values, dim=1).item()
