from controller import Supervisor
import numpy as np
import random
import socket
import json

class CustomCarEnv:
    robot = Supervisor()
    def __init__(self):
    
        
        self.timestep = int(self.robot.getBasicTimeStep())
        
        '''
        for i in range(self.robot.getNumberOfDevices()):
            device = self.robot.getDeviceByIndex(i)
            print(f"Device {i}: name = {device.getName()}, type = {device.getNodeType()}")
        '''

        self.left_motor = self.robot.getDevice('left_rear_wheel')
        self.right_motor = self.robot.getDevice('right_rear_wheel')

        self.front_left_steer = self.robot.getDevice('left_steer')
        self.front_right_steer = self.robot.getDevice('right_steer')

        if self.front_left_steer is None or self.front_right_steer is None:
            print("ERRORE: Attuatori di sterzo non trovati.")
            exit()

        #=====Gestione lidars=====
        self.lidar_front = self.robot.getDevice('lidar_front')
        self.lidar_front.enable(self.timestep)
        self.lidar_front.enablePointCloud() #attiva la nuvola di punti
        
        self.lidar_rear = self.robot.getDevice('lidar_rear')
        if self.lidar_rear is not None:
            self.lidar_rear.enable(self.timestep)
            self.lidar_rear.enablePointCloud()
        else:
            print("AVVISO: Lidar posteriore non trovato.")

        self.collision_th = 0.5
        #==========================

        self.max_timesteps = 5000 #settato a 1000 per 100m di percorso
        self.curr_timestep = 0
        self.curr_episode = 0

        #=====Road settings=====
        self.road_length = 520
        self.road_width = 8

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

        '''
        self.target_x = 59.4
        self.target_y = 0
        self.target_z = 0.12
        '''
        
        self.target_node = self.robot.getFromDef('target')
        self.target_translation = self.target_node.getField('translation')
        self.target_pos = self.target_translation.getSFVec3f()
        self.distance_target_threshold = 2
        print(f'Coordinate target: {self.target_pos}') #DEBUG

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

        self.total_reward = 0.0
        self.reward_print_interval = 50  # ogni 50 step stampa
        #=========================

        #=====Oggetti da randomizzare (ostacoli, barili ecc.)=====
        self.spawn_range_x = (0, self.road_length)
        self.spawn_range_y = (-self.road_width/2, self.road_width/2)

        self.random_objects = []
        i=0

        while True:
            node = self.robot.getFromDef(f'ostacolo_{i}')
            if not node:
                break
            self.random_objects.append({
                    'node': node,
                    'translation': node.getField('translation'),
                    'rotation':node.getField('rotation')
            })
            print(f'coordinate ostacolo {i}: {node.getField('translation').getSFVec3f()}') #DEBUG
            i += 1
        self.num_obst = i-1
        #print(f'Numero ostacoli trovati: {self.num_obst}') DEBUG
        #========================

        #======Anti-Blockage=====
        self.block_counter = 0
        self.max_block_steps = 50
        self.last_pos = None
        self.block_movement_threshold = 0.01  # distanza minima per considerare movimento
        self.min_speed_threshold = 0.05       # velocità media sotto la quale la macchina è considerata quasi ferma


        self.udr_called = False


        self.reset()

    def step(self, action):
    
        avg_speed = 0.5 * (action[0] + action[1])
        self.left_motor.setVelocity(avg_speed)
        self.right_motor.setVelocity(avg_speed)

        # Imposta l'angolo di sterzata delle ruote anteriori
        self.front_left_steer.setPosition(action[2])
        self.front_right_steer.setPosition(action[2])

        #print(f"STEP: azione ricevuta = {action}")  # DEBUG
        
        if self.robot.step(self.timestep) == -1:
            print("Simulazione interrotta durante lo step.")  # DEBUG
            return None, 0.0, True, {}

        obs = self._get_obs()
        reward = self._compute_reward(obs)

        self.total_reward += reward

        if self.curr_timestep % self.reward_print_interval == 0:
            print(f"[{self.curr_timestep}] Reward cumulativa episodio {self.curr_episode}: {self.total_reward:.2f}")


        done, cause = self._check_done(obs)
        self.curr_timestep += 1
        #print(f'current step: {self.curr_timestep}') #DEBUG
        #print(f"STEP: obs = {obs}, reward = {reward:.4f}, done = {done}")  # DEBUG

        if done:
            print(f"[FINE][{cause}]Episodio {self.curr_episode} terminato con reward cumulativa: {self.total_reward:.2f}\n")
            print("==========================")
            self.curr_episode += 1
            self.udr_called = False
            obs = self.reset()
            return obs, reward, True, {}


        return obs, reward, False, {}

    def _get_obs(self):
        #=====Gestione velocità=====
        left_velocity = np.array([self.left_motor.getVelocity()], dtype=np.float32)

        left_velocity = (left_velocity - 40.0) / 90.0 #Normalization

        #print(f'left_v: {left_velocity}') #DEBUG
        #print(f'l_velocity shape: {left_velocity.shape}') #DEBUG

        right_velocity = np.array([self.right_motor.getVelocity()], dtype=np.float32)

        right_velocity = (right_velocity - 40.0) / 90.0 #Normalization

        #print(f'right_v: {right_velocity}') #DEBUG
        #print(f'r_velocity shape: {right_velocity.shape}') #DEBUG
        #===========================

        #=====Gestione posizione gps=====
        pos = np.array(self.gps.getValues() if self.gps else [0.0, 0.0, 0.0], dtype=np.float32)

        pos[0] /= self.road_length
        pos[1] /= self.road_width
        pos[2] /= 1.0

        #print(f'pos: {pos}') #DEBUG
        #print(f'pos shape: {len(pos)}') #DEBUG
        #================================

        #=====Gestione ribaltamento=====
        max_angle = 0.25 #15 gradi
        orientation = np.array(self.imu.getRollPitchYaw() if self.imu else [0.0, 0.0, 0.0], dtype=np.float32)

        orientation[0] /= max_angle
        orientation[1] /= max_angle
        orientation[2] /= np.pi

        #print(f'orientation: {orientation}') #DEBUG
        #print(f'orientation shape: {len(orientation)}') #DEBUG
        #===============================

        ##=====Gestione rotazione=====
        rotation = np.array(self.rotation_field.getSFVec3f(), dtype=np.float32)

        rotation[3] /= np.pi

        #print(f'rot: {rotation}') #DEBUG
        #print(f'rotation shape: {rotation.shape}') #DEBUG
        #============================

        ##=====Gestione lidars=====
        lidar_front_values = self.lidar_front.getRangeImage() #array di distanze
        #===DEBUG===
        lidar_front_values = np.array(self.lidar_front.getRangeImage(), dtype=np.float32)

        # Filtra i valori inf sostituendoli con il max range del lidar
        lidar_front_values[np.isinf(lidar_front_values)] = self.lidar_front.getMaxRange()
        #print(f'tot_front_lidar shape: {lidar_front_values.shape}') #DEBUG
        
        num_samples = 10
        step = max(1, len(lidar_front_values) // num_samples)

        lidar_front_samples = lidar_front_values[::step][:num_samples]
        lidar_front_samples = lidar_front_samples / self.lidar_front.getMaxRange() #normalization

        #print(f'distanza lidar front: {lidar_front_samples}')
        #print(f'lidar shape: {lidar_front_samples.shape}') #DEBUG
        
        lidar_rear_values = np.array(self.lidar_rear.getRangeImage(), dtype=np.float32)

        lidar_rear_values[np.isinf(lidar_rear_values)] = self.lidar_rear.getMaxRange()

        lidar_rear_samples = lidar_rear_values[::step][:num_samples]
        lidar_rear_samples = lidar_rear_samples / self.lidar_rear.getMaxRange() #Normalization
        #print(f'distanza lidar rear: {lidar_rear_samples}')
        #print(f'lidar rear shape: {lidar_rear_samples.shape}') #DEBUG
        #==========================================================

        obs_space = np.concatenate(
            [
            left_velocity, right_velocity,
            pos,
            rotation,
            lidar_front_samples,
            lidar_rear_samples,
            orientation
            ], dtype=np.float32)
        #print(f'obs_space shape: {obs_space.shape}') #DEBUG
        return obs_space

    def _compute_reward(self, obs):
        # === Parse observation ===
        left_v = obs[0]
        right_v = obs[1]
        pos = obs[2:5]
        roll, pitch, yaw = obs[-3], obs[-2], obs[-1]
        lidar_front_samples = obs[9:19]
        lidar_rear_samples = obs[19:29]

        # === Distanza attuale dal target ===
        current_distance = np.linalg.norm(np.array([
            self.target_pos[0] - (pos[0] * self.road_length), #Denormalization
            self.target_pos[1]- (pos[1] * self.road_width),
            self.target_pos[2] - (pos[2] * 1.0)
        ]))
        prev_distance = getattr(self, 'prev_distance', None)

        # === Reward per progresso verso il target ===
        progress_reward = 0.0
        if prev_distance is not None:
            progress_reward = prev_distance - current_distance
        self.prev_distance = current_distance

        # === Reward per vicinanza precisa al target (entro 1m) ===
        proximity_reward = np.exp(-current_distance)

        # === Penalità collisione netta ===
        normalized_th = self.collision_th / self.lidar_front.getMaxRange()
        front_collision_penalty = -1.0 if np.any(lidar_front_samples < normalized_th) else 0.0
        rear_collision_penalty = -1.0 if np.any(lidar_rear_samples < normalized_th) else 0.0


        # === Penalità ribaltamento ===
        fall_penalty = -1.0 if abs(roll) > 0.2 or abs(pitch) > 0.2 else 0.0

        # === Penalità sterzata brusca (penalizza angoli sterzata grandi, ma non troppo) ===
        steer_left = self.front_left_steer.getTargetPosition()
        steer_right = self.front_right_steer.getTargetPosition()
        avg_steer = 0.5 * (steer_left + steer_right)
        steer_penalty = -0.01 * (avg_steer ** 2)  # penalità quadratica più dolce

        # === Penalità per velocità differenziale fra ruote (zig-zag) ===
        velocity_penalty = -0.1 * abs(left_v - right_v)

        #===Penalità vicinanza a ostacoli frontali (media distanza lidar frontale)===
        front_lidar_penalty = 0.0
        mean_front_lidar = np.mean(lidar_front_samples)
        if mean_front_lidar < normalized_th * 3:
            front_lidar_penalty = -0.5 * (normalized_th * 3 - mean_front_lidar) / (normalized_th * 3)

        

        rear_lidar_penalty = 0.0
        mean_rear_lidar = np.mean(lidar_rear_samples)
        if mean_rear_lidar < normalized_th * 3:
            rear_lidar_penalty = -0.5 * (normalized_th * 3 - mean_rear_lidar) / (normalized_th * 3)

        # === Bonus retromarcia lenta ===
        avg_speed = 0.5 * (left_v + right_v)
        reverse_bonus = 0.1 if avg_speed < 0 else 0.0

        # === Penalità tempo (episodi lunghi) ===
        time_penalty = -0.0005 * self.curr_timestep

        # === Bonus finale raggiungimento target ===
        target_bonus = 10.0 if current_distance < self.distance_target_threshold else 0.0

        # === Reward totale ===
        reward = (
            2.0 * progress_reward
            + 3.0 * proximity_reward
            + reverse_bonus
            + front_collision_penalty
            + rear_collision_penalty
            + fall_penalty
            + steer_penalty
            + velocity_penalty
            + front_lidar_penalty
            + rear_lidar_penalty
            + target_bonus
            + time_penalty
        )

        return np.clip(reward, -5.0, 5.0)

    def _check_done(self, obs):
       
        cause = None

        # Controllo urto: almeno uno dei 5 valori più piccoli deve essere sotto soglia
        front_lidar_dist = obs[9:19]
        rear_lidar_dist = obs[19:29]

        normalized_th = self.collision_th/self.lidar_front.getMaxRange()

        #print(last_5_lidar) #DEBUG
        collision = np.any(front_lidar_dist < normalized_th) or np.any(rear_lidar_dist < normalized_th)
        if collision:
            cause = 'collision'
        #print(f'collision: {collision}') #DEBUG

        # Controllo timeout
        timeout = self.curr_timestep >= self.max_timesteps
        if timeout and cause is None:
            cause = 'timeout'
        #print(f'timeout: {timeout}') #DEBUG

        #controllo target
        #print(f'posizione del target: {self.target_pos}')
        tesla_x = obs[2] * self.road_length #Denormalization
        tesla_y = obs[3] * self.road_width
        tesla_z = obs[4] * 1.0
        target_distance = np.sqrt((self.target_pos[0] - tesla_x)**2 +\
                                  (self.target_pos[1] - tesla_y)**2 +\
                                    (self.target_pos[2] - tesla_z)**2  )
        if target_distance < self.distance_target_threshold and cause is None:
            cause = 'target_reached'

        #print(f'target: {target_distance < self.distance_target_threshold}') #DEBUG
        
        #controllo caduta(ribaltamento)
        roll, pitch = obs[-3], obs[-2]
        falling = abs(roll) > 0.2 or abs(pitch) > 0.2 
        if falling and cause is None:
            cause = 'falling'
        #print(f'flipped: {falling}, roll: {roll:.2f}, pitch: {pitch:.2f}')  # DEBUG

        # === Anti-block system ===
        curr_pos = np.array([obs[2]*self.road_length, obs[3]*self.road_width, obs[4] * 1.0])  # posizione GPS corrente
        avg_speed = 0.5 * ((90 * obs[0] + 40) + (90 * obs[1] + 40))  # media velocità ruote

        if self.last_pos is not None:
            delta_movement = np.linalg.norm(curr_pos - self.last_pos)

            if delta_movement < self.block_movement_threshold or abs(avg_speed) < self.min_speed_threshold:
                self.block_counter += 1
            else:
                self.block_counter = 0
        self.last_pos = curr_pos

        is_blocked = self.block_counter > self.max_block_steps
        '''
        if is_blocked:
            print(f"[ANTI-BLOCK] Macchina bloccata per {self.block_counter} step consecutivi.")
        '''
        if is_blocked and cause is None:
            cause = 'blocked'
        
        done = collision or timeout or target_distance < self.distance_target_threshold or falling or is_blocked

        return done, cause

    def reset(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.front_left_steer.setPosition(0.0)
        self.front_right_steer.setPosition(0.0)

        self.prev_distance = None
        
        self.curr_timestep = 0

        self.total_reward = 0.0

        self.block_counter = 0
        self.last_pos = None

        
        if not self.udr_called:
            self.udr()
            self.target_pos = self.target_translation.getSFVec3f()
            self.udr_called = True

        self.car_node.setVelocity([0, 0, 0, 0, 0, 0]) #reset fisico totale della macchina
        self.car_node.resetPhysics()

        #self.translation_field.setSFVec3f([0.502863, 0.493051, 0.446674])
        #self.rotation_field.setSFRotation([0.0791655, -0.995765, 0.0467393,0.0157421])

        
        for _ in range(20):
            self.robot.step(self.timestep)
        

        return self._get_obs()

    def assign_objects_and_target(self, min_distance=3.0, num_obj=7):
        # Definizione dei rettilinei: start, end, coordinata costante, asse costante
        straight_sections = [
            {'start': 0, 'end': 120, 'const': 0, 'axis': 'y'},      # rettilineo 1
            {'start': 180, 'end': 260, 'const': -20, 'axis': 'y'}, # rettilineo 2
            {'start': 320, 'end': 400, 'const': 0, 'axis': 'y'},  # rettilineo 3
            {'start': 460, 'end': 520, 'const': 20, 'axis': 'y'}    # rettilineo 4
        ]

        if len(self.random_objects) < num_obj:
            print(f"Errore: servono almeno {num_obj} oggetti random per posizionare {num_obj} ostacoli.")
            return

        # Prendi ostacoli casuali
        random.shuffle(self.random_objects)
        selected_obstacles = self.random_objects[:num_obj]

        # Aggiungi il target
        all_objects = selected_obstacles + [self.target]
        random.shuffle(all_objects)  # mescolali per distribuirli in modo casuale

        for i, section in enumerate(straight_sections):
            # Prendi due oggetti per questo rettilineo
            section_objects = all_objects[i*2 : (i+1)*2]
            placed_coords = []

            for obj in section_objects:
                node = obj['node']
                translation_field = obj['translation']
                rotation_field = obj['rotation']

                # Trova una coordinata che non sia troppo vicina alle altre
                attempts = 0
                while True:
                    coord = random.uniform(section['start'], section['end'])
                    const_offset = random.uniform((-self.road_width / 2)-0.25, (self.road_width / 2)-0.25)

                    if all(abs(coord - other) >= min_distance for other in placed_coords):
                        placed_coords.append(coord)
                        break

                    attempts += 1
                    if attempts > 100:
                        print(f"Impossibile posizionare {node.getDef()} nel rettilineo {i+1}")
                        break

                # Costruisci la nuova posizione
                if section['axis'] == 'x':
                    pos = [section['const'] + const_offset, coord, 0.4]
                else:
                    pos = [coord, section['const'] + const_offset, 0.4]

                # Assegna posizione e rotazione
                translation_field.setSFVec3f(pos)
                rotation_field.setSFRotation([0, 0, -1, 0])

                print(f"{node.getDef()} posizionato in {pos} (rettilineo {i+1}), rotazione settata")

    def udr(self):
        #=====Posizione iniziale random della macchina=====
        rand_x = self.default_car_pos[0] + np.random.uniform(-3, 5)
        rand_y = np.random.uniform(self.spawn_range_y[0] + 1, self.spawn_range_y[1] - 1)  # leggermente dentro i bordi strada
        self.translation_field.setSFVec3f([rand_x, rand_y, 0.7])

        #=====Rotazione iniziale random della macchina=====
        angle = np.random.uniform(-np.pi, np.pi)
        self.rotation_field.setSFRotation([0, 0, 1, angle])  # ruota intorno a z

        #=====Posizioniamo oggetti e target nel mondo=====
        self.assign_objects_and_target(3, self.num_obst+1)



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