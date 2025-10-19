from pacman_env import PacmanEnv
from train_dqn_pacman import DQN
import torch
import numpy as np
from Pacman import run

env = PacmanEnv()
state_dim = len(env.reset())
action_dim = env.action_space

model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("dqn_pacman500.pth"))
model.eval()

state = env.reset()
done = False

while not done:
    with torch.no_grad():
        action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()
    

    next_state, reward, done = env.step(action)
    state = next_state
