import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from pacman_env import PacmanEnv

# ======================
# 1. Rede Neural DQN
# ======================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ======================
# 2. Funções auxiliares
# ======================
def select_action(state, epsilon, policy_net, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    with torch.no_grad():
        q_values = policy_net(torch.tensor(state, dtype=torch.float32))
        return q_values.argmax().item()

def replay(memory, batch_size, gamma, policy_net, target_net, optimizer):
    """Treina o DQN com minibatch do replay buffer."""
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
    expected_q = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.functional.mse_loss(q_values, expected_q.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ======================
# 3. Hiperparâmetros
# ======================


# ======================
# 4. Inicialização
# ======================
def train():
    episodes = 2500
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9995
    batch_size = 64
    lr = 1e-3
    target_update = 10
    memory_size = 50000

    env = PacmanEnv()
    state_dim = len(env.reset())
    action_dim = env.action_space

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    memory = deque(maxlen=memory_size)

    i = 0
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:

            action = select_action(state, epsilon, policy_net, action_dim)

            next_state, reward, done = env.step(action)
            total_reward += reward

            memory.append((state, action, reward, next_state, done))

            replay(memory, batch_size, gamma, policy_net, target_net, optimizer)
            state = next_state

        # Atualiza rede alvo
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Ep {episode+1:03d} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.2f}")

        # if i % 500 == 0:
        #     torch.save(policy_net.state_dict(), f"dqn_pacman{i}.pth")
        # i += 1
    
    torch.save(policy_net.state_dict(), f"dqn_pacman{i}.pth")

    print("Treinamento finalizado ✅")

    


# train()