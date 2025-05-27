from controller import Supervisor
import numpy as np
import socket
import json
import time

class CustomCarEnv:
    def __init__(self):
    
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        '''
        for i in range(self.robot.getNumberOfDevices()):
            device = self.robot.getDeviceByIndex(i)
            print(f"Device {i}: name = {device.getName()}, type = {device.getNodeType()}")
        '''

        self.left_motor = self.robot.getDevice('left_rear_wheel')
        self.right_motor = self.robot.getDevice('right_rear_wheel')

        self.lidar = self.robot.getDevice('lidar_front')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud() #attiva la nuvola di punti
        
        self.collision_th = 0.2
        self.max_timesteps = 1000 #settato per 100m di percorso
        self.curr_timestep = 0
        self.curr_episode = 0

        # Supervisor: ottieni riferimento al nodo Tesla
        self.car_node = self.robot.getFromDef("tesla3")
        self.translation_field = self.car_node.getField('translation')
        self.rotation_field = self.car_node.getField('rotation')

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

        
        self.reset()

    def step(self, action):
    
        left_speed, right_speed = action
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        #print(f"STEP: azione ricevuta = {action}")  # DEBUG
        
        if self.robot.step(self.timestep) == -1:
            print("Simulazione interrotta durante lo step.")  # DEBUG
            return None, 0.0, True, {}

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self._check_done(obs)
        self.curr_timestep += 1
        print(f'current step: {self.curr_timestep}') #DEBUG
        #print(f"STEP: obs = {obs}, reward = {reward:.4f}, done = {done}")  # DEBUG

        if done:
            print("Episodio terminato: collisione o ribaltamento o timeout.")  # DEBUG
            obs = self.reset()
            return obs, reward, True, {}

        return obs, reward, done, {}

    def _get_obs(self):
        left_velocity = np.array([self.left_motor.getVelocity()], dtype=np.float32)
        #print(f'left_v: {left_velocity}') #DEBUG

        right_velocity = np.array([self.right_motor.getVelocity()], dtype=np.float32)
        #print(f'right_v: {right_velocity}') #DEBUG

        pos = self.gps.getValues() if self.gps else [0.0, 0.0, 0.0]
        #print(f'pos: {pos}') #DEBUG

        rotation = np.array(self.rotation_field.getSFVec3f(), dtype=np.float32)
        #print(f'rot: {rotation}') #DEBUG

        lidar_values = self.lidar.getRangeImage() #array di distanze
        #===DEBUG===
        lidar_values = np.array(self.lidar.getRangeImage(), dtype=np.float32)

        # Filtra i valori inf sostituendoli con il max range del lidar (fallback sensato)
        lidar_values[np.isinf(lidar_values)] = self.lidar.getMaxRange()
        lidar_values.sort()
        top_5_smallest_distances = lidar_values[:5] #valid_distances[:5] 
        print(f'distanza: {top_5_smallest_distances}')
        #======

        return np.concatenate([left_velocity, right_velocity, pos, rotation, top_5_smallest_distances], dtype=np.float32)

    def _compute_reward(self, obs, action):
    
        forward_velocity = obs[0]
        reward = forward_velocity - 1e-3 * np.square(action).sum()
        return reward

    def _check_done(self, obs):
       
        # Controllo urto: tutti e 5 i valori più piccoli devono essere sotto soglia
        last_5_lidar = obs[-5:]
        collision = all(d < self.collision_th for d in last_5_lidar)
        print(f'collision: {collision}') #DEBUG

        # Controllo timeout
        timeout = self.curr_timestep >= self.max_timesteps
        print(f'timeout: {timeout}') #DEBUG

        return collision or timeout


    def reset(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        print(f'current_episode: {self.curr_episode}') #DEBUG
        self.curr_episode+=1
        self.curr_timestep = 0

        # Reset posizione e orientamento
        self.translation_field.setSFVec3f([0.5, 0, 0.3])  # <-- aggiorna se serve
        self.rotation_field.setSFRotation([0, 1, 0, 0])        # <-- reset YAW a 0
        self.robot.step(self.timestep)  # applica il reset

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
