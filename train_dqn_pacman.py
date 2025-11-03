import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from pacman_env import PacmanEnv
import csv


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


def select_action(state, epsilon, policy_net, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    with torch.no_grad():
        q_values = policy_net(torch.tensor(state, dtype=torch.float32))
        return q_values.argmax().item()

def replay(memory, batch_size, gamma, policy_net, target_net, optimizer):
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.from_numpy(np.array(states, dtype=np.float32)) #mudei pra n dar warning
    actions = torch.tensor(actions).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
    expected_q = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.functional.mse_loss(q_values, expected_q.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train():
    # mudei alguns desses parametros pra testar, mas acho q deu quase no mesmo, nem vou botar no artigo
    episodes = 2500
    gamma = 0.95 # antes 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.999 # antes 0.9995
    batch_size = 128 # antes 64
    lr = 5e-4 # antes 1e-4
    target_update = 10
    memory_size = 50000
    model_save_interval = 500

    env = PacmanEnv()
    state_dim = len(env.reset())
    action_dim = env.action_space

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    memory = deque(maxlen=memory_size)

    #imprime o csv para os gráficos
    log_file = "training_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "epsilon", "loss"])

    print("treinamento iniciado")

    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            episode_loss = []

            while not done:
                action = select_action(state, epsilon, policy_net, action_dim)
                next_state, reward, done = env.step(action)
                total_reward += reward

                memory.append((state, action, reward, next_state, done))
                loss = replay(memory, batch_size, gamma, policy_net, target_net, optimizer)
                if loss is not None:
                    episode_loss.append(loss)

                state = next_state

            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            avg_loss = np.mean(episode_loss) if episode_loss else 0.0 #tem alguma coisa errada e sempre entra no else

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode + 1, total_reward, epsilon, avg_loss]) #todos os avg_loss são 0.0 por conta daquilo

            print(f"Ep {episode + 1:04d} | Reward: {total_reward:7.2f} | "
                  f"Epsilon: {epsilon:.3f} | Loss: {avg_loss:.5f}")

            if episode % model_save_interval == 0 and episode > 0:
                torch.save(policy_net.state_dict(), f"dqn_pacman_{episode}.pth")

    #salva o arquivo mesmo q fechando o programa antes de terminar
    except KeyboardInterrupt:
        torch.save(policy_net.state_dict(), f"dqn_pacman_parcial.pth")
        print("\nmodelo salvo")

    finally:
        torch.save(policy_net.state_dict(), f"dqn_pacman_final.pth")
        print("treinamento finalizado")


if __name__ == "__main__":
    train()