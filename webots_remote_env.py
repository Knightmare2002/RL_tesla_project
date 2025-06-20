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

        # Corrected observation space shape based on _get_obs concatenation in CustomCarEnv:
        # combined_velocity (1) + pos (3) + orientation (3) + front_lidar_samples (10) + rear_lidar_samples (10) + target_coords_normalized (3) = 30
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        '''
        OBSERVATION SPACE (Corrected Mapping from CustomCarEnv's _get_obs)
        0   : combined_velocity_norm
        1-3 : pos_x, pos_y, pos_z _norm
        4-6 : roll, pitch, yaw _norm
        7-9: target_x, target_y, target_z _norm
        10-19: front_lidar_samples (10 values) _norm
        '''

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]), 
            high=np.array([1.0, 1.0]),   #[eccelerazione, rotazione]
            dtype=np.float32
        )

    
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