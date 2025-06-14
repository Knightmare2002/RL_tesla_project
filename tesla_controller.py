from controller import Supervisor
from scipy.interpolate import CubicSpline
import numpy as np
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

        self.max_timesteps = 2000 #settato a 1000 per 100m di percorso
        self.curr_timestep = 0
        self.curr_episode = 0

        #=====Road settings=====
        self.road_length = 200
        self.road_width = 10

        #=====Supervisor: ottieni riferimento al nodo Tesla======
        self.car_node = self.robot.getFromDef("tesla3")
        self.translation_field = self.car_node.getField('translation')
        self.rotation_field = self.car_node.getField('rotation')

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

        self.min_dist_obj = 3.0
        self.min_num_obst = 12

        self.random_objects = []
        i=0

        while True:
            node = self.robot.getFromDef(f'ostacolo_{i}')
            if not node:
                break
            self.random_objects.append({
                    'node': node,
                    'translation': node.getField('translation')
            })
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


        done = self._check_done(obs)
        self.curr_timestep += 1
        #print(f'current step: {self.curr_timestep}') #DEBUG
        #print(f"STEP: obs = {obs}, reward = {reward:.4f}, done = {done}")  # DEBUG

        if done:
            print(f"[FINE] Episodio {self.curr_episode} terminato con reward cumulativa: {self.total_reward:.2f}\n")
            print("==========================")
            self.curr_episode += 1
            self.udr_called = False
            obs = self.reset()
            return obs, reward, True, {}


        return obs, reward, False, {}

    def _get_obs(self):
        #=====Gestione velocità=====
        left_velocity = np.array([self.left_motor.getVelocity()], dtype=np.float32)
        #print(f'left_v: {left_velocity}') #DEBUG
        #print(f'l_velocity shape: {left_velocity.shape}') #DEBUG

        right_velocity = np.array([self.right_motor.getVelocity()], dtype=np.float32)
        #print(f'right_v: {right_velocity}') #DEBUG
        #print(f'r_velocity shape: {right_velocity.shape}') #DEBUG
        #===========================

        #=====Gestione posizione gps=====
        pos = self.gps.getValues() if self.gps else [0.0, 0.0, 0.0]
        #print(f'pos: {pos}') #DEBUG
        #print(f'pos shape: {len(pos)}') #DEBUG
        #================================

        #=====Gestione ribaltamento=====
        orientation = self.imu.getRollPitchYaw() if self.imu else [0.0, 0.0, 0.0]
        #print(f'orientation: {orientation}') #DEBUG
        #print(f'orientation shape: {len(orientation)}') #DEBUG
        #===============================

        ##=====Gestione rotazione=====
        rotation = np.array(self.rotation_field.getSFVec3f(), dtype=np.float32)
        #print(f'rot: {rotation}') #DEBUG
        #print(f'rotation shape: {rotation.shape}') #DEBUG
        #============================

        ##=====Gestione lidars=====
        lidar_front_values = self.lidar_front.getRangeImage() #array di distanze
        #===DEBUG===
        lidar_front_values = np.array(self.lidar_front.getRangeImage(), dtype=np.float32)

        # Filtra i valori inf sostituendoli con il max range del lidar
        lidar_front_values[np.isinf(lidar_front_values)] = self.lidar_front.getMaxRange()
        lidar_front_values.sort()
        front_lidar_smallest_distance = np.array([lidar_front_values[0]])
        #print(f'distanza: {front_lidar_smallest_distance}')
        #print(f'lidar shape: {front_lidar_smallest_distance.shape}') #DEBUG
        
        lidar_rear_values = np.array(self.lidar_rear.getRangeImage(), dtype=np.float32) if self.lidar_rear else np.array([self.lidar.getMaxRange()] * len(self.lidar.getRangeImage()), dtype=np.float32)
        lidar_rear_values[np.isinf(lidar_rear_values)] = self.lidar_rear.getMaxRange() if self.lidar_rear else self.lidar.getMaxRange()
        lidar_rear_values.sort()
        rear_lidar_smallest_distance = np.array([lidar_rear_values[0]])


        obs_space = np.concatenate([left_velocity, right_velocity, pos, rotation, front_lidar_smallest_distance, rear_lidar_smallest_distance, orientation], dtype=np.float32)
        #print(f'obs_space shape: {obs_space.shape}') #DEBUG
        return obs_space

    def _compute_reward(self, obs):
        # === Parse observation ===
        left_v = obs[0]
        right_v = obs[1]
        pos = obs[2:5]
        roll, pitch, yaw = obs[11:14]
        front_lidar_val = obs[9]
        rear_lidar_val = obs[10]

        # === Distanza attuale dal target ===
        current_distance = np.linalg.norm(np.array([
            self.target_pos[0] - pos[0],
            self.target_pos[1]- pos[1],
            self.target_pos[2] - pos[2]
        ]))
        prev_distance = getattr(self, 'prev_distance', None)

        # === Reward per progresso verso il target ===
        progress_reward = 0.0
        if prev_distance is not None:
            progress_reward = prev_distance - current_distance
        self.prev_distance = current_distance

        # === Reward per vicinanza precisa al target (entro 1m) ===
        proximity_reward = 0.0
        if current_distance < 1.0:
            proximity_reward = 1.0 - current_distance

        # === Penalità collisione netta ===
        front_collision_penalty = -1.0 if front_lidar_val < self.collision_th else 0.0

        rear_collision_penalty = -1.0 if rear_lidar_val < self.collision_th else 0.0

        # === Penalità ribaltamento ===
        fall_penalty = -1.0 if abs(roll) > 0.15 or abs(pitch) > 0.15 else 0.0

        # === Penalità sterzata brusca (penalizza angoli sterzata grandi, ma non troppo) ===
        steer_left = self.front_left_steer.getTargetPosition()
        steer_right = self.front_right_steer.getTargetPosition()
        avg_steer = 0.5 * (steer_left + steer_right)
        steer_penalty = -0.01 * (avg_steer ** 2)  # penalità quadratica più dolce

        # === Penalità per velocità differenziale fra ruote (zig-zag) ===
        velocity_penalty = -0.1 * abs(left_v - right_v)

        # === Penalità proporzionale alla distanza minima lidar ===
        # se è vicino a ostacoli, penalizzo proporzionalmente, altrimenti 0
        front_lidar_penalty = 0.0
        if front_lidar_val < self.collision_th * 3:
            # ad esempio penalità lineare decrescente da 0 a -0.5 tra collision_th*3 e 0
            front_lidar_penalty = -0.5 * (self.collision_th * 3 - front_lidar_val) / (self.collision_th * 3)
        
        rear_lidar_penalty = 0.0
        if rear_lidar_val < self.collision_th * 3:
            rear_lidar_penalty = -0.5 * (self.collision_th * 3 - rear_lidar_val) / (self.collision_th * 3)

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
       
        # Controllo urto: almeno uno dei 5 valori più piccoli deve essere sotto soglia
        front_lidar_dist = obs[9]
        rear_lidar_dist = obs[10]
        #print(last_5_lidar) #DEBUG
        collision = front_lidar_dist < self.collision_th or rear_lidar_dist < self.collision_th
        #print(f'collision: {collision}') #DEBUG

        # Controllo timeout
        timeout = self.curr_timestep >= self.max_timesteps
        #print(f'timeout: {timeout}') #DEBUG

        #controllo target
        #print(f'posizione del target: {self.target_pos}')
        tesla_x = obs[2]
        tesla_y = obs[3]
        tesla_z = obs[4]
        target_distance = np.sqrt((self.target_pos[0] - tesla_x)**2 +\
                                  (self.target_pos[1] - tesla_y)**2 +\
                                    (self.target_pos[2] - tesla_z)**2  )
        #print(f'target: {target_distance < self.distance_target_threshold}') #DEBUG
        
        #controllo caduta(ribaltamento)
        roll, pitch = obs[11], obs[12]
        falling = abs(roll) > 0.05 or abs(pitch) > 0.05  # circa 30°
        #print(f'flipped: {falling}, roll: {roll:.2f}, pitch: {pitch:.2f}')  # DEBUG

        # === Anti-block system ===
        curr_pos = np.array([obs[2], obs[3], obs[4]])  # posizione GPS corrente
        avg_speed = 0.5 * (obs[0] + obs[1])  # media velocità ruote

        if self.last_pos is not None:
            delta_movement = np.linalg.norm(curr_pos - self.last_pos)

            if delta_movement < self.block_movement_threshold and abs(avg_speed) > self.min_speed_threshold:
                self.block_counter += 1
            else:
                self.block_counter = 0
        self.last_pos = curr_pos

        is_blocked = self.block_counter > self.max_block_steps
        if is_blocked:
            print(f"[ANTI-BLOCK] Macchina bloccata per {self.block_counter} step consecutivi.")

        
        return collision or timeout or target_distance < self.distance_target_threshold or falling or is_blocked

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

    def resample_path(self, points, resolution=1.0):
        resampled = []
        for i in range(len(points) - 1):
            p0 = np.array(points[i])
            p1 = np.array(points[i+1])
            segment_vec = p1 - p0
            segment_len = np.linalg.norm(segment_vec)
            n_steps = max(int(segment_len / resolution), 1)

            for j in range(n_steps):
                t = j / n_steps
                interp_point = (1 - t) * p0 + t * p1
                resampled.append(tuple(interp_point))
        resampled.append(points[-1])  # include last point
        return resampled

    def place_objects_randomly(self):
        #=====Ottieni waypoints 2D (x,y)=====
        road_node = self.robot.getFromDef('road')
        waypoints_field = road_node.getField('wayPoints')
        n_points = waypoints_field.getCount()
        points = []
        for i in range(n_points):
            wp = waypoints_field.getMFVec3f(i)
            points.append((wp[0], wp[1]))
        
        resampled_points = self.resample_path(points, 0.25) #dobbiamo aumentare il numero di waypoints, definiamo ogni 0.25 metri
        #print(f'numero punti interpolati: {len(resampled_points)}') DEBUG

        # Interpolazione spline per avere traiettoria continua e derivata
        points = np.array(resampled_points)
        distances = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        distances = np.insert(distances, 0, 0)  # inserisci 0 all'inizio
        
        cs_x = CubicSpline(distances, points[:, 0])
        cs_y = CubicSpline(distances, points[:, 1])
        path_length = distances[-1]

        placed_positions = []  # Lista posizioni (x,y) ostacoli e target
        
        def sample_position(min_dist_from_car=10.0, min_dist_between=7.0):
            max_attempts = 1000
            for _ in range(max_attempts):
                # Campiona una posizione casuale lungo la lunghezza della strada
                s = np.random.uniform(0, path_length)
                center_x = cs_x(s)
                center_y = cs_y(s)

                # Calcola tangente (derivata prima)
                dx = cs_x(s, 1)
                dy = cs_y(s, 1)
                tangent = np.array([dx, dy])
                tangent /= np.linalg.norm(tangent)

                # Calcola la normale (perpendicolare al tangente)
                normal = np.array([-tangent[1], tangent[0]])

                # Scegli uno spostamento laterale random entro i limiti della larghezza strada
                lateral_offset = np.random.uniform(-self.road_width/2 + 0.5, self.road_width/2 - 0.5)  # 0.5 margine per sicurezza

                pos = np.array([center_x, center_y]) + lateral_offset * normal

                # Controllo distanza minima da macchina
                car_pos = np.array(self.gps.getValues()[:2]) if self.gps else np.array([0,0])
                dist_from_car = np.linalg.norm(pos - car_pos)
                if dist_from_car < min_dist_from_car:
                    continue

                # Controllo distanza da ostacoli e target già piazzati
                if any(np.linalg.norm(pos - p) < min_dist_between for p in placed_positions):
                    continue

                return pos
            raise RuntimeError("Non sono riuscito a piazzare un oggetto rispettando le distanze minime")

        # Posiziona target
        target_pos = sample_position(min_dist_from_car=10.0, min_dist_between=5.0)
        self.target_translation.setSFVec3f([target_pos[0], target_pos[1], self.target_translation.getSFVec3f()[2]])
        placed_positions.append(target_pos)

        # Posiziona ostacoli
        for obj in self.random_objects:
            obst_pos = sample_position(min_dist_from_car=10.0, min_dist_between=5.0)
            obj['translation'].setSFVec3f([obst_pos[0], obst_pos[1], obj['translation'].getSFVec3f()[2]])
            placed_positions.append(obst_pos)

    def udr(self):
        #=====Posizione iniziale random della macchina=====
        rand_x = np.random.uniform(0, 5)
        rand_y = np.random.uniform(self.spawn_range_y[0] + 1, self.spawn_range_y[1] - 1)  # leggermente dentro i bordi strada
        self.translation_field.setSFVec3f([rand_x, rand_y, 0.7])

        #=====Rotazione iniziale random della macchina=====
        angle = np.random.uniform(-np.pi, np.pi)
        self.rotation_field.setSFRotation([0, 0, 1, angle])  # ruota intorno a z

        #=====Posizioniamo oggetti e target nel mondo=====
        self.place_objects_randomly()



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