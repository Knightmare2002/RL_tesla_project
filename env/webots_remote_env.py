import numpy as np
import socket
import json
import gymnasium as gym
from gymnasium import spaces

class WebotsRemoteEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.host = '127.0.0.1'
        self.port = 10000
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.host, self.port))

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)

    def step(self, action):
        msg = json.dumps({'cmd': 'step', 'action': action.tolist()}).encode()
        self.conn.send(msg)
        response = self.conn.recv(1024)
        data = json.loads(response.decode())
        obs = np.array(data['obs'], dtype=np.float32)
        reward = data['reward']
        done = data['done']
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.conn.send(json.dumps({'cmd': 'reset'}).encode())
        response = self.conn.recv(1024)
        data = json.loads(response.decode())
        obs = np.array(data['obs'], dtype=np.float32)
        return obs, {}

    def close(self):
        self.conn.send(json.dumps({'cmd': 'exit'}).encode())
        self.conn.close()
