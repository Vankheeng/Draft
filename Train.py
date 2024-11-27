import argparse
import os
import shutil
from random import random, randint, sample
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from CustomAgent import DeepQNetwork
# from TetrisBattle.envs.tetris_interface import TetrisSingleInterface
from TetrisBattle.envs.tetris_env import TetrisSingleEnv

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=17, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    # parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=1, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt, model, optimizer, start_epoch=0):
    epoch = start_epoch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if not os.path.exists(opt.saved_path):
        os.makedirs(opt.saved_path)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="rgb_array")
    criterion = nn.MSELoss()
    # sẽ đổi thành q value
    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()
    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    while epoch < opt.num_epochs:
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        if random_action:
            action = env.random_action()
        else:
            model.eval()
            with torch.no_grad():
                recent_state = env.get_seen_grid()
                recent_state = torch.tensor(recent_state, dtype=torch.float32).unsqueeze(0)  # Tạo batch dimension
                print(recent_state.shape)
                if torch.cuda.is_available():
                    recent_state = recent_state.cuda()
                predictions = model(recent_state)
                action = torch.argmax(predictions).item()
        # if torch.cuda.is_available():
        #     next_state = next_state.cuda()
        next_state, reward, done, info = env.step(action)
        replay_memory.append([next_state, reward, next_state, done])
        if done:
            # final_score = env.score
            # final_tetrominoes = env.tetrominoes
            # final_cleared_lines = env.cleared_lines
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        for idx, state in enumerate(state_batch):
            print(f"State {idx} shape: {state.shape}")
        state_batch = torch.stack([torch.tensor(state, dtype=torch.float32) for state in state_batch])
        reward_batch = torch.tensor(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack([torch.tensor(state, dtype=torch.float32) for state in next_state_batch])

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Action: {}".format(
            epoch,
            opt.num_epochs,
            action))
            # final_score,
            # final_tetrominoes,
            # final_cleared_lines))
        # writer.add_scalar('Train/Score', final_score, epoch - 1)
        # writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        # writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))
        if epoch > 0 and epoch % opt.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, filepath=checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")

    save_checkpoint(model, optimizer, epoch, filepath=checkpoint_path)
    torch.save(model, "{}/tetris".format(opt.saved_path))

# Lưu checkpoint
def save_checkpoint(model, optimizer, replay_memory, epoch, filepath='model_checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'replay_memory': list(replay_memory),  # Chuyển deque thành list để lưu
    }, filepath)

def load_checkpoint(model, optimizer, replay_memory, filepath='model_checkpoint.pth'):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    replay_memory.extend(checkpoint['replay_memory'])  # Khôi phục replay_memory
    epoch = checkpoint['epoch']
    return epoch


# if __name__ == "__main__":
#     opt = get_args()
#     train(opt)
#     load_checkpoint()
if __name__ == "__main__":
    opt = get_args()

    # Kiểm tra xem checkpoint có tồn tại không
    checkpoint_path = "model_checkpoint.pth"
    model = DeepQNetwork()  # Tạo model
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)  # Tạo optimizer

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    # Bắt đầu training từ epoch đã lưu
    train(opt, model, optimizer, start_epoch)