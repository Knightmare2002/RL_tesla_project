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

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
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
        9   lidar_1
        10  lidar_2
        11  lidar_3
        12  lidar_4
        13  lidar_5
        14  roll
        15  pitch
        16  yawn
        '''

        self.action_space = spaces.Box(
            low=np.array([-50.0, -50.0, -0.5]),   # aggiunta sterzata (circa -30°)
            high=np.array([130.0, 130.0, 0.5]),   # circa +30°
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