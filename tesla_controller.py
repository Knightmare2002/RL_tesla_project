from controller import Robot
import numpy as np
import socket
import json
import time

class CustomCarEnv:
    def __init__(self):
    
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        '''
        for i in range(self.robot.getNumberOfDevices()):
            device = self.robot.getDeviceByIndex(i)
            print(f"Device {i}: name = {device.getName()}, type = {device.getNodeType()}")
        '''

        self.left_motor = self.robot.getDevice('left_rear_wheel')
        self.right_motor = self.robot.getDevice('right_rear_wheel')



        if self.left_motor is None or self.right_motor is None:
            print("ERRORE: Motori non trovati. Assicurati di usare un nodo basato su 'Car' come TeslaModel3.")
            exit()

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Sensori
        self.gps = self.robot.getDevice('gps')
        if self.gps is not None:
            self.gps.enable(self.timestep)
        else:
            print("AVVISO: GPS non trovato.")

        self.imu = self.robot.getDevice('inertial unit')
        if self.imu is not None:
            self.imu.enable(self.timestep)
        else:
            print("AVVISO: Inertial Unit non trovata.")
        
        print("Inizializzazione completata.")  # DEBUG

        self.reset()

    def step(self, action):
    
        left_speed, right_speed = action
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        print(f"STEP: azione ricevuta = {action}")  # DEBUG
        
        if self.robot.step(self.timestep) == -1:
            print("Simulazione interrotta durante lo step.")  # DEBUG
            return None, 0.0, True, {}

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self._check_done(obs)
        print(f"STEP: obs = {obs}, reward = {reward:.4f}, done = {done}")  # DEBUG

        return obs, reward, done, {}

    def _get_obs(self):
    
        pos = [0.0, 0.0, 0.0]
        if self.gps:
            pos = self.gps.getValues()

        orientation = [0.0, 0.0, 0.0]
        if self.imu:
            orientation = self.imu.getRollPitchYaw()

        return np.array(pos + orientation, dtype=np.float32)

    def _compute_reward(self, obs, action):
    
        forward_velocity = obs[0]
        reward = forward_velocity - 1e-3 * np.square(action).sum()
        return reward

    def _check_done(self, obs):
    
        roll, pitch, yaw = obs[3], obs[4], obs[5]
        return abs(roll) > 1.0 or abs(pitch) > 1.0

    def reset(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        for _ in range(10):
            if self.robot.step(self.timestep) == -1:
                print("Simulazione interrotta durante il reset.") #DEBUG
                return np.zeros(6, dtype=np.float32)
        
        return self._get_obs()


# --- Socket server per comunicazione RL esterna ---
HOST = '127.0.0.1'
PORT = 10000

env = CustomCarEnv()

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print("Controller Webots in ascolto sulla porta", PORT, "...")

        conn, addr = s.accept()
        with conn:
            print(f"Connesso a: {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    print("Client disconnesso.")
                    break

                try:
                    msg = json.loads(data.decode())
                except json.JSONDecodeError:
                    print(f"Errore di decodifica JSON: {data.decode()}")
                    continue

                if msg['cmd'] == 'reset':
                    obs = env.reset()
                    conn.send(json.dumps({'obs': obs.tolist()}).encode())

                elif msg['cmd'] == 'step':
                    obs, reward, done, _ = env.step(msg['action'])
                    conn.send(json.dumps({
                        'obs': obs.tolist(),
                        'reward': float(reward),
                        'done': bool(done)
                    }).encode())

                elif msg['cmd'] == 'exit':
                    print("Comando 'exit' ricevuto.")
                    break

except Exception as e:
    print(f"Errore nel server socket: {e}")
finally:
    print("Chiusura del controller Webots.")
