from controller import Supervisor
import numpy as np
import socket
import json

#TODO INSERISCI UN SENSORE POSTERIORE PER LA RETROMARCIA IDENTICO A QUELLO FRONTALE

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

        self.lidar = self.robot.getDevice('lidar_front')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud() #attiva la nuvola di punti
        
        self.collision_th = 0.5
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

        self.target_x = 59.4
        self.target_y = 0
        self.target_z = 0.12
        
        self.target_node = self.robot.getFromDef('target')
        self.target_translation = self.target_node.getField('translation')
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
        left_velocity = np.array([self.left_motor.getVelocity()], dtype=np.float32)
        #print(f'left_v: {left_velocity}') #DEBUG

        right_velocity = np.array([self.right_motor.getVelocity()], dtype=np.float32)
        #print(f'right_v: {right_velocity}') #DEBUG

        pos = self.gps.getValues() if self.gps else [0.0, 0.0, 0.0]
        #print(f'pos: {pos}') #DEBUG

        orientation = self.imu.getRollPitchYaw() if self.imu else [0.0, 0.0, 0.0]
        #print(f'orientation: {orientation}') #DEBUG

        rotation = np.array(self.rotation_field.getSFVec3f(), dtype=np.float32)
        #print(f'rot: {rotation}') #DEBUG

        lidar_values = self.lidar.getRangeImage() #array di distanze
        #===DEBUG===
        lidar_values = np.array(self.lidar.getRangeImage(), dtype=np.float32)

        # Filtra i valori inf sostituendoli con il max range del lidar
        lidar_values[np.isinf(lidar_values)] = self.lidar.getMaxRange()
        lidar_values.sort()
        top_5_smallest_distances = lidar_values[:5] #valid_distances[:5] 
        #print(f'distanza: {top_5_smallest_distances}')
        #======

        return np.concatenate([left_velocity, right_velocity, pos, rotation, top_5_smallest_distances, orientation], dtype=np.float32)

    def _compute_reward(self, obs):
        # === Parse observation ===
        left_v = obs[0]
        right_v = obs[1]
        pos = obs[2:5]
        roll, pitch, yaw = obs[14:17]
        lidar_vals = obs[9:14]

        # === Distanza attuale dal target ===
        current_distance = np.linalg.norm(np.array([
            self.target_x - pos[0],
            self.target_y - pos[1],
            self.target_z - pos[2]
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
        collision_penalty = -1.0 if np.any(lidar_vals < self.collision_th) else 0.0

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
        min_lidar_dist = np.min(lidar_vals)
        # se è vicino a ostacoli, penalizzo proporzionalmente, altrimenti 0
        lidar_penalty = 0.0
        if min_lidar_dist < self.collision_th * 3:
            # ad esempio penalità lineare decrescente da 0 a -0.5 tra collision_th*3 e 0
            lidar_penalty = -0.5 * (self.collision_th * 3 - min_lidar_dist) / (self.collision_th * 3)

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
            + collision_penalty
            + fall_penalty
            + steer_penalty
            + velocity_penalty
            + lidar_penalty
            + target_bonus
            + time_penalty
        )

        return np.clip(reward, -5.0, 5.0)

    def _check_done(self, obs):
       
        # Controllo urto: tutti e 5 i valori più piccoli devono essere sotto soglia
        last_5_lidar = obs[-5:]
        collision = all(d < self.collision_th for d in last_5_lidar)
        #print(f'collision: {collision}') #DEBUG

        # Controllo timeout
        timeout = self.curr_timestep >= self.max_timesteps
        #print(f'timeout: {timeout}') #DEBUG

        #controllo target
        tesla_x = obs[2]
        tesla_y = obs[3]
        tesla_z = obs[4]
        target_distance = np.sqrt((self.target_x - tesla_x)**2 +\
                                  (self.target_y - tesla_y)**2 +\
                                    (self.target_z - tesla_z)**2  )
        #print(f'target: {target_distance < self.distance_target_threshold}') #DEBUG
        
        #controllo caduta(ribaltamento)
        roll, pitch = obs[14], obs[15]
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
#
        
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
            self.udr_called = True

        self.car_node.setVelocity([0, 0, 0, 0, 0, 0]) #reset fisico totale della macchina
        self.car_node.resetPhysics()

        #self.translation_field.setSFVec3f([0.502863, 0.493051, 0.446674])
        #self.rotation_field.setSFRotation([0.0791655, -0.995765, 0.0467393,0.0157421])

        
        for _ in range(20):
            self.robot.step(self.timestep)
        

        return self._get_obs()

    def udr(self):
            #=====Posizione iniziale random della macchina=====
            rand_x = np.random.uniform(0, 5)
            rand_y = np.random.uniform(self.spawn_range_y[0]+1, self.spawn_range_y[1]-1)  # leggermente dentro i bordi strada
            self.translation_field.setSFVec3f([rand_x, rand_y, 0.7])

            #=====Posizione iniziale random del target=====
            # Cambia range x in modo che il target sia vicino alla fine strada (es. ultimi 10m)
            self.target_x = np.random.uniform(self.road_length - 10, self.road_length - 5)
            self.target_y = np.random.uniform(self.spawn_range_y[0]+1, self.spawn_range_y[1]-1)
            self.target_translation.setSFVec3f([self.target_x, self.target_y, 0.12])  # z bassa per essere sul terreno


            #=====Rotazione iniziale random della macchina=====
            angle = np.random.uniform(-np.pi, np.pi)
            self.rotation_field.setSFRotation([0, 0, 1, angle]) #ruota intro z

            #===== Randomizzazione con distanza minima tra ostacoli =====
            range_x = [10, self.road_length - 10]  # ostacoli posizionati dopo i primi 10m e prima degli ultimi 10m
            range_y = [self.spawn_range_y[0]+1, self.spawn_range_y[1]-1]
            z = 0.4
            min_dist = self.min_dist_obj

            placed_positions = []

            # Includiamo la posizione della macchina e del target per evitare sovrapposizioni
            car_pos = np.array([rand_x, rand_y])
            target_pos = np.array([self.target_x, self.target_y])
            placed_positions.append(car_pos)
            placed_positions.append(target_pos)

            # Controlla quanti ostacoli abbiamo nel mondo e quanti ne servono almeno
            num_objects = len(self.random_objects)
            num_obstacles_to_place = max(num_objects, self.min_num_obst)

            # Se il numero di ostacoli esistenti è minore di min_num_obstacles, avvisa
            if num_objects < self.min_num_obst:
                print(f"[AVVISO] Numero ostacoli disponibili ({num_objects}) < numero minimo consigliato ({self.min_num_obst}).")

            # Ciclo per posizionare ostacoli, minimo num_obstacles_to_place
            for i in range(num_obstacles_to_place):
                # Se ci sono meno ostacoli nel mondo dei posti da posizionare, ricicliamo da random_objects o creiamo logica extra (qui semplice riciclo)
                obj = self.random_objects[i % num_objects]

                is_position_valid = False
                max_attempts = 200  # aumentato per maggiore chance
                attempt = 0

                while not is_position_valid and attempt < max_attempts:
                    new_x = np.random.uniform(*range_x)
                    new_y = np.random.uniform(*range_y)
                    new_position = np.array([new_x, new_y])

                    # Controllo distanza minima da tutte le posizioni già piazzate
                    is_overlapping = any(
                        np.linalg.norm(new_position - pos) < min_dist
                        for pos in placed_positions
                    )

                    if not is_overlapping:
                        is_position_valid = True
                        obj['translation'].setSFVec3f([new_x, new_y, z])
                        placed_positions.append(new_position)
                    attempt += 1

                if attempt == max_attempts:
                    print(f"[AVVISO] Impossibile posizionare ostacolo dopo {max_attempts} tentativi.")



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