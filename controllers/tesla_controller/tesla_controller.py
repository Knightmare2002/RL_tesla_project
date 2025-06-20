from controller import Supervisor
import numpy as np
import socket
import json
import os


class CustomCarEnv:
    robot = Supervisor()

    def _open_debug_file(self):
        """Apre il file di debug, sovrascrivendolo se esiste già."""
        try:
            self.debug_file = open(self.debug_file_path, 'w')
            self.debug_file.write(f"--- Inizio sessione di debug Webots RL ---\n")
        except IOError as e:
            print(f"Errore nell'apertura del file di debug: {e}")
            self.debug_file = None # Assicura che sia None se c'è un errore

    def _close_debug_file(self):
        """Chiude il file di debug se è aperto."""
        if self.debug_file:
            self.debug_file.write(f"--- Fine sessione di debug ---\n")
            self.debug_file.close()
            self.debug_file = None

    def __init__(self):
        self.timestep = int(self.robot.getBasicTimeStep())


        #===== Gestione Motori====
        self.left_motor = self.robot.getDevice('left_rear_wheel')
        self.right_motor = self.robot.getDevice('right_rear_wheel')

        self.front_left_steer = self.robot.getDevice('left_steer')
        self.front_right_steer = self.robot.getDevice('right_steer')

        self.max_speed = 26 #rad/s
        self.max_back_speed = 0 #rad/s

        self.max_steering = 0.6 #rad

        self.norm_max = 1
        self.norm_min = 0

        if self.front_left_steer is None or self.front_right_steer is None:
            print("ERRORE: Attuatori di sterzo non trovati.")
            exit()

        #=====Gestione lidars=====
        self.lidar_front = self.robot.getDevice('lidar_front')

        if self.lidar_front is not None:
            self.lidar_front.enable(self.timestep)
            self.lidar_front.enablePointCloud()
        else:
            print("AVVISO: Lidar frontale non trovato.")
        
        #print(f'Max range lidar: {self.lidar_front.getMaxRange()}') DEBUG

        self.collision_th = 1.0
        #==========================

        self.max_timesteps = 2000 # Adjusted for a shorter 100m track
        self.curr_timestep = 0
        self.curr_episode = 0

        #=====Road settings=====
        self.road_length = 120 # Changed to 100m straight line
        self.road_width = 12

        #=====Supervisor: ottieni riferimento al nodo Tesla======
        self.car_node = self.robot.getFromDef("tesla3")
        self.translation_field = self.car_node.getField('translation')
        self.rotation_field = self.car_node.getField('rotation')
        self.default_car_pos = self.translation_field.getSFVec3f()

        if self.left_motor is None or self.right_motor is None:
            print("ERRORE: Motori non trovati.")
            exit()

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.front_left_steer.setPosition(0.0)
        self.front_right_steer.setPosition(0.0)

        self.target_node = self.robot.getFromDef('target')
        self.target_translation = self.target_node.getField('translation')
        self.target_pos = self.target_translation.getSFVec3f()

        self.distance_target_threshold = 2.5

        print(f'Coordinate target iniziali: {self.target_pos}') #DEBUG

        self.target = {
            'node':self.target_node,
            'translation': self.target_translation,
            'rotation':self.target_node.getField('rotation')
        }
        #==================

        #=====Sensori=====
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
        self.max_angle_roll_pitch_yaw = np.pi
        #========================

        #=====Reward=====
        self.total_reward = 0.0
        self.reward_print_interval = 50   # ogni 50 step stampa
        #=========================

        #======Anti-Blockage=====
        self.block_counter = 0
        self.max_block_steps = 50
        self.last_pos = None
        self.block_movement_threshold = 0.01   # distanza minima per considerare movimento
        self.min_speed_threshold = 0.05    # velocità media sotto la quale la macchina è considerata quasi ferma

        # --- Setup per il file di debug ---
        self.debug_file_path = 'debug.txt'
        self.debug_file = None
        self._open_debug_file()
        # ----------------------------------

        self.reset() # Initial reset to set up the environment

    def step(self, action):

        avg_speed = action[0] * self.max_speed
        self.left_motor.setVelocity(avg_speed)
        self.right_motor.setVelocity(avg_speed)

        # Imposta l'angolo di sterzata delle ruote anteriori
        self.front_left_steer.setPosition(action[1] * self.max_steering) 
        self.front_right_steer.setPosition(action[1] * self.max_steering)

        if self.robot.step(self.timestep) == -1:
            print("Simulazione interrotta durante lo step.")
            return None, 0.0, True, {}

        obs = self._get_obs()
        reward = self._compute_reward(obs)

        self.total_reward += reward

        if self.curr_timestep % self.reward_print_interval == 0:
            print(f"[{self.curr_timestep}] Reward cumulativa episodio {self.curr_episode}: {self.total_reward:.2f}")

        done, cause = self._check_done(obs)
        self.curr_timestep += 1

        if done:
            print(f"[FINE][{cause}]Episodio {self.curr_episode} terminato con reward cumulativa: {self.total_reward:.2f}\n")
            print("==========================")
            
            self.curr_episode += 1
            obs = self.reset() # The reset happens here
            return obs, reward, True, {}

        return obs, reward, False, {}

    def _get_obs(self):
        #=====Gestione velocità=====
        left_velocity = self.left_motor.getVelocity()
        right_velocity = self.right_motor.getVelocity()
        avg_velocity = 0.5 * (left_velocity + right_velocity)

        avg_velocity_norm = avg_velocity / self.max_speed

        avg_velocity_norm = np.array([avg_velocity_norm], dtype=np.float32)

        #=====Gestione posizione gps=====
        pos = np.array(self.gps.getValues() if self.gps else [0.0, 0.0, 0.0], dtype=np.float32)
        
        pos_norm = np.copy(pos)

        # Normalizzazione GPS per observation space
        pos_norm[0] /= self.road_length
        pos_norm[1] /= self.road_width/2


        #=====Gestione ribaltamento IMU (Roll, Pitch, Yaw)=====
        imu = np.array(self.imu.getRollPitchYaw() if self.imu else [0.0, 0.0, 0.0], dtype=np.float32)
        

        # Normalizzazione degli angoli in radianti per observation space
        #Roll rotazione intorno x
        #Pitch rotazione intorno z
        #Yaw rotazione intorno y

        orientation_obs_norm = np.copy(imu)
        orientation_obs_norm[0] /= self.max_angle_roll_pitch_yaw # Roll
        orientation_obs_norm[1] /= self.max_angle_roll_pitch_yaw/2 # Pitch
        orientation_obs_norm[2] /= self.max_angle_roll_pitch_yaw        # Yaw

        
        #=====Gestione lidars=====
        lidar_front_values = np.array(self.lidar_front.getRangeImage(), dtype=np.float32)

        #DEBUG
        #print(f"DEBUG: Max Lidar Range del sensore (da Webots): {self.lidar_front.getMaxRange()} m")
        #print(f"DEBUG: Lunghezza Raw Lidar Values: {len(lidar_front_values)}")
        #print(f"DEBUG: Raw Lidar Values (prima dell'inf handling): {lidar_front_values[:20]} ... {lidar_front_values[-20:]}") # Stampa l'inizio e la fine{lidar_front_samples_norm}')

        lidar_front_values[np.isinf(lidar_front_values)] = self.lidar_front.getMaxRange()
        # DEBUG
        #print(f"DEBUG: Lidar Values after Inf handling (prima del campionamento): {lidar_front_values[:20]} ... {lidar_front_values[-20:]}")

        num_samples = 10
        step = max(1, len(lidar_front_values) // num_samples)
        lidar_front_samples_obs = lidar_front_values[::step][:num_samples]
        lidar_front_samples_norm = lidar_front_samples_obs / self.lidar_front.getMaxRange() #normalization
        
        #DEBUG
        #print(f'Obs space lidars distances (sampled): {lidar_front_samples_obs}')
        #print(f'Obs space lidars distances norm (sampled): {lidar_front_samples_norm}')


        #==========================================================

        #======Gestione posizione del target======
        target_coords = np.array(self.target['translation'].getSFVec3f(), dtype=np.float32)

        target_coords_norm = np.copy(target_coords)

        target_coords_norm[0] /= self.road_length
        target_coords_norm[1] /= (self.road_width / 2)
        target_coords_norm[2] /= 1.0

        #print(f'Posizione target obs: {target_coords}...{target_coords_norm}_norm') #DEBUG

        # Concatenazione delle osservazioni
        obs_space = np.concatenate(
            [
            avg_velocity_norm,                     #1 valore (obs[0])
            pos_norm,                          # 3 valori (obs[1:3])
            orientation_obs_norm,                  # 3 valori (obs[4:6])
            target_coords_norm,          # 3 valori (obs[17:19])
            lidar_front_samples_norm,          # 10 valori (obs[7:16])
            ], dtype=np.float32)
        return obs_space

    def _compute_reward(self, obs):
        reward = 0.0
        
        # Inizializza i componenti della reward per il debug
        reward_components = {
            'progress_reward': 0.0,
            'target_reached_reward': 0.0,
            'collision_penalty': 0.0,
            'falling_penalty': 0.0,
            'blocked_penalty': 0.0,
            'time_penalty': 0.0,
            'proximity_penalty': 0.0
        }

        avg_speed = obs[0] *self.max_speed
        
        tesla_pos_norm = obs[1:4]
        tesla_pos = np.copy(tesla_pos_norm)
        tesla_pos[0] *= self.road_length
        tesla_pos[1] *= self.road_width

        orientation_norm = obs[4:7]
        orientation = np.copy(orientation_norm) * self.max_angle_roll_pitch_yaw

        target_norm = obs[7:10]
        target = np.copy(target_norm)
        target[0] *= self.road_length
        target[1] *= self.road_width

        lidars_norm = obs[10:20]
        lidars = np.copy(lidars_norm) * self.lidar_front.getMaxRange()


        current_distance_from_target = np.linalg.norm(target - tesla_pos)


        # 1. Progress Reward
        if self.prev_distance is not None:
            distance_reduction = self.prev_distance - current_distance_from_target
            reward_components['progress_reward'] = distance_reduction * 0.5
        self.prev_distance = current_distance_from_target

        # 2. Target Reached Reward
        if current_distance_from_target < self.distance_target_threshold:
            reward_components['target_reached_reward'] = 100.0

        # 3. Collision Penalty
        min_distance_lidar = np.min(lidars)
        collision = (min_distance_lidar < self.collision_th and avg_speed> 0.1)
        
        if collision:
            reward_components['collision_penalty'] = -10.0

        # 4. Falling Penalty
        #Roll rotazione intorno x
        #Pitch rotazione intorno y
        #Yaw rotazione intorno z
        max_angle_for_fall = 0.2
        falling = abs(orientation[0]) > max_angle_for_fall or abs(orientation[1]) > max_angle_for_fall
        if falling:
            reward_components['falling_penalty'] = -9.0

        # 5. Blocked Penalty
        if self.block_counter > 0:
            reward_components['blocked_penalty'] = -np.exp(0.03 * self.block_counter)

        '''
        # 6. Speed Reward
        ideal_speed_min = 10.0
        ideal_speed_max = 26.0

        if avg_speed >= ideal_speed_min and avg_speed <= ideal_speed_max:
            reward_components['speed_reward'] = 0.1 * (avg_speed / ideal_speed_max)
        elif avg_speed > ideal_speed_max:
            reward_components['speed_reward'] = -0.05 * (avg_speed - ideal_speed_max)
        elif avg_speed < ideal_speed_min and avg_speed > 1.0:
            reward_components['speed_reward'] = 0.05 * (avg_speed / ideal_speed_min)
        else:
            reward_components['speed_reward'] = -0.2
        
        # 7. Off-Road Penalty
        road_half_width = self.road_width / 2
        car_on_road = abs(tesla_pos[1]) < road_half_width

        if not car_on_road:
            reward_components['off_road_penalty'] = -1.0
        '''
        # 8. Time Penalty
        reward_components['time_penalty'] = -0.05

        #9. Proximity penalty
        proximity_warning_dist_front = 1.5 # metri, es. penalizza se a meno di 1.5 metri

        if not collision: # Only apply proximity penalty if not already in collision
            if min_distance_lidar < proximity_warning_dist_front:
                proximity = proximity_warning_dist_front - min_distance_lidar
                reward_components['proximity_penalty'] += -np.exp(proximity * 2)


        # Calcola la reward totale prima del clipping
        raw_reward = sum(reward_components.values())

        # Applica il clipping alla reward finale
        reward = np.clip(raw_reward, -10.0, 10.0)

        return reward
    
    def _check_done(self, obs):

        avg_speed = obs[0] *self.max_speed

        tesla_pos_norm = obs[1:4]
        tesla_pos = np.copy(tesla_pos_norm)
        tesla_pos[0] *= self.road_length
        tesla_pos[1] *= self.road_width

        orientation_norm = obs[4:7]
        orientation = np.copy(orientation_norm) * self.max_angle_roll_pitch_yaw

        target_norm = obs[7:10]
        target = np.copy(target_norm)
        target[0] *= self.road_length
        target[1] *= self.road_width
        #print(f'posizione target end: {target}') DEBUG

        lidars_norm = obs[10:20]
        lidars = np.copy(lidars_norm) * self.lidar_front.getMaxRange()


        current_distance_from_target = np.linalg.norm(target - tesla_pos)
        #print(f'distance from target: {tesla_pos} -> {target} = {current_distance_from_target}\n') #DEBUG

        cause = None

        #===== Controllo urto =====
        #DEBUG
        #print(f"Collision lidar distances: {lidars}") #DEBUG
        #print(f'Collision min lidar distance: {np.min(lidars)}') #DEBUG
        collision = np.min(lidars) < self.collision_th
        if collision:
            cause = 'collision'

        #===== Controllo timeout =====
        timeout = self.curr_timestep >= self.max_timesteps
        if timeout and cause is None:
            cause = 'timeout'

        # controllo target

        #target_coords = self.target['translation'].getSFVec3f()

        target_reached = current_distance_from_target < self.distance_target_threshold
        if target_reached and cause is None:
            cause = 'target_reached'

        # controllo caduta(ribaltamento)
        max_angle_for_fall = 0.2 
        falling = abs(orientation[0]) > max_angle_for_fall or abs(orientation[1]) > max_angle_for_fall
        if falling and cause is None:
            cause = 'falling'

        # === Anti-block system ===

        if self.last_pos is not None:
            delta_movement = np.linalg.norm(tesla_pos - self.last_pos)

            if delta_movement < self.block_movement_threshold or abs(avg_speed) < self.min_speed_threshold:
                self.block_counter += 1
            else:
                self.block_counter = 0
        self.last_pos = tesla_pos

        is_blocked = self.block_counter > self.max_block_steps
        if is_blocked and cause is None:
            cause = 'blocked'

        done = collision or timeout or target_reached or falling or is_blocked

        return done, cause

    def reset(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.front_left_steer.setPosition(0.0)
        self.front_right_steer.setPosition(0.0)

        self.translation_field.setSFVec3f(self.default_car_pos)
        self.rotation_field.setSFRotation([0, 1, 0, 0])

        self.prev_distance = None

        self.curr_timestep = 0
        self.total_reward = 0.0
        self.block_counter = 0
        self.last_pos = None

        # Directly call setup function, curriculum learning removed
        #self._setup_environment_for_episode(num_obstacles=2) 

        self.car_node.setVelocity([0, 0, 0, 0, 0, 0]) #reset fisico totale della macchina
        self.car_node.resetPhysics()

        for _ in range(20): # Piccola pausa per la stabilizzazione del simulatore
            self.robot.step(self.timestep)

        print(f"Inizio Episodio {self.curr_episode}") 
        
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
                    env.robot.simulationQuit(0)
                    break

except Exception as e:
    print(f"Errore nel server socket: {e}")

finally:
    print("Chiusura del controller Webots.")