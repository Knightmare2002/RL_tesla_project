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

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)
        '''
        OBSERVATION SPACE
        0   left_velocity
        1   right_velocity
        2   pos_x
        3   pos_y
        4   pos_z
        5   rot_1
        6   rot_2
        7   rot_3
        8   rot_4
        9-18   front_lidar
        19-28  rear_lidar
        29  roll
        30  pitch
        31  yawn
        '''

        self.action_space = spaces.Box(
            low=np.array([-50.0, -50.0, -0.5]),   # aggiunta sterzata (circa -30°)
            high=np.array([130.0, 130.0, 0.5]),   # circa +30°
            dtype=np.float32
        )

    def normalize_obs(self, obs):
        obs = np.copy(obs)

        # Velocità ruote
        obs[0:2] = (obs[0:2] - 40.0) / 90.0  # da [-50,130] → [-1,1]

        # Posizione X ∈ [0,60] → [-1,1]
        obs[2] = (obs[2] - 30.0) / 30.0
        obs[3] = obs[3] / 3.5                # Y ∈ [-3.5,3.5] → [-1,1]
        obs[4] = (obs[4] - 0.12) / 0.12      # Z ∈ ~[0,0.24]

        # Rotazioni
        obs[5:9] = np.clip(obs[5:9], -1, 1)  # already likely normalized

        
        # Orientamenti (roll, pitch, yaw) ∈ [-π, π]
        obs[29:32] = obs[29:32] / np.pi

        return np.clip(obs, -1.0, 1.0)

    def step(self, action):
        msg = json.dumps({'cmd': 'step', 'action': action.tolist()}).encode()
        self.conn.send(msg)
        response = self.conn.recv(1024)
        data = json.loads(response.decode())
        obs = np.array(data['obs'], dtype=np.float32)
        reward = data['reward']
        done = data['done']
        norm_obs = self.normalize_obs(obs)
        return norm_obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.conn.send(json.dumps({'cmd': 'reset'}).encode())
        response = self.conn.recv(1024)
        data = json.loads(response.decode())
        obs = np.array(data['obs'], dtype=np.float32)
        norm_obs = self.normalize_obs(obs)
        return norm_obs, {}

    def close(self):
        self.conn.send(json.dumps({'cmd': 'exit'}).encode())
        self.conn.close()