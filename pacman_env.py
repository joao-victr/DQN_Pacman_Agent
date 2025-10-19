import numpy as np
import pygame
from Pacman import game, run, reset, running

class PacmanEnv:
    def __init__(self):
        self.action_space = 4 
        self.prev_score = 0

    def reset(self):

        reset()
        game.gameOver = False
        game.lives = 1
        self.prev_score = 0
        state = self._get_state()
        return state

    def step(self, action):

        global running
        pygame.event.pump()

        if not running:
            running = True
        run(action)
        

        reward = game.reward
        # print(reward)
        game.reward = 0
        self.prev_score = game.score

        done = game.gameOver
        if done:
            print(done)

        next_state = self._get_state()

        return next_state, reward, done

    def _get_state(self):
        pac = game.pacman
        ghosts = game.ghosts

        state = np.array([
            pac.mouthOpen,
            pac.dir,
            pac.row / len(game.gameBoard), pac.col / len(game.gameBoard[0]),
            ghosts[0].row / len(game.gameBoard), ghosts[0].col / len(game.gameBoard[0]),
            ghosts[1].row / len(game.gameBoard), ghosts[1].col / len(game.gameBoard[0]),
            ghosts[2].row / len(game.gameBoard), ghosts[2].col / len(game.gameBoard[0]),
            ghosts[3].row / len(game.gameBoard), ghosts[3].col / len(game.gameBoard[0]),
        ], dtype=np.float32)

        return state

    def render(self):
        pygame.display.update()
