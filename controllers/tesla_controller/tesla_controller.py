from controller import Supervisor
import numpy as np
import random
import socket
import json

class CustomCarEnv:
    robot = Supervisor()

    def __init__(self):
        self.timestep = int(self.robot.getBasicTimeStep())

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

        self.max_timesteps = 5000
        self.curr_timestep = 0
        self.curr_episode = 0

        #=====Road settings=====
        self.road_length = 260
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

        self.target_node = self.robot.getFromDef('target')
        self.target_translation = self.target_node.getField('translation')
        self.target_pos = self.target_translation.getSFVec3f()
        self.distance_target_threshold = 2
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

        self.total_reward = 0.0
        self.reward_print_interval = 50  # ogni 50 step stampa
        #=========================

        #=====Oggetti da randomizzare (ostacoli, barili ecc.)=====
        self.spawn_range_x = (0, self.road_length)
        self.spawn_range_y = (-self.road_width/4, self.road_width/4)

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
            i += 1
        self.num_total_obstacles = i 
        #print(f'Numero ostacoli trovati: {self.num_total_obstacles}') # DEBUG
        #========================

        #======Anti-Blockage=====
        self.block_counter = 0
        self.max_block_steps = 50
        self.last_pos = None
        self.block_movement_threshold = 0.01  # distanza minima per considerare movimento
        self.min_speed_threshold = 0.05       # velocità media sotto la quale la macchina è considerata quasi ferma

        self.udr_called = False

        # --- Curriculum Learning Parameters ---
        self.curriculum_stage = 0 # Inizia dalla fase più semplice
        self.stage_settings = {
            0: {'target_x_range': (15, 60), 'num_obstacles_to_use': 2, 'reward_threshold_to_advance': 30.0},
            1: {'target_x_range': (65, 120), 'num_obstacles_to_use': 4, 'reward_threshold_to_advance': 50.0},
            2: {'target_x_range': (125, 200), 'num_obstacles_to_use': 6, 'reward_threshold_to_advance': 70.0},
            3: {'target_x_range': (205, self.road_length), 'num_obstacles_to_use': self.num_total_obstacles, 'reward_threshold_to_advance': 90.0}
        }
        self.consecutive_successes = 0
        self.num_successes_to_advance = 5 # Numero di episodi di successo consecutivi per avanzare

        self.reset()

    def step(self, action):

        avg_speed = action[0] 
        self.left_motor.setVelocity(avg_speed)
        self.right_motor.setVelocity(avg_speed)

        # Imposta l'angolo di sterzata delle ruote anteriori
        self.front_left_steer.setPosition(action[1]) 
        self.front_right_steer.setPosition(action[1])

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
            
            current_stage_threshold = self.stage_settings[self.curriculum_stage]['reward_threshold_to_advance']

            # Verifica se l'episodio è un vero "successo" per il curriculum
            if cause == 'target_reached' and self.total_reward >= current_stage_threshold:
                self.consecutive_successes += 1
                print(f"Successo consecutivo: {self.consecutive_successes}/{self.num_successes_to_advance}")
                if self.consecutive_successes >= self.num_successes_to_advance and self.curriculum_stage < len(self.stage_settings) - 1:
                    self.curriculum_stage += 1
                    self.consecutive_successes = 0 # Resetta per la nuova fase (dopo l'avanzamento)
                    print(f"*** AVANZAMENTO CURRICULUM: Nuova fase {self.curriculum_stage} ***")
            else:
                # Resetta i successi consecutivi SOLO SE l'episodio NON è un successo
                self.consecutive_successes = 0 
                # Aggiungi un messaggio per chiarire il reset
                if cause != 'target_reached':
                    print(f"Episodio terminato per '{cause}', successi consecutivi resettati.")
                elif self.total_reward < current_stage_threshold:
                    print(f"Reward insufficiente ({self.total_reward:.2f} < {current_stage_threshold:.2f}), successi consecutivi resettati.")


            self.curr_episode += 1
            self.udr_called = False # Permette una nuova UDR al prossimo reset
            obs = self.reset() # Il reset avviene qui
            return obs, reward, True, {}

        return obs, reward, False, {}

    def _get_obs(self):
        #=====Gestione velocità=====
        left_velocity = np.array([self.left_motor.getVelocity()], dtype=np.float32)
        left_velocity = (left_velocity - 40.0) / 90.0 #Normalization

        right_velocity = np.array([self.right_motor.getVelocity()], dtype=np.float32)
        right_velocity = (right_velocity - 40.0) / 90.0 #Normalization

        #=====Gestione posizione gps=====
        pos = np.array(self.gps.getValues() if self.gps else [0.0, 0.0, 0.0], dtype=np.float32)
        # Normalizzazione GPS
        pos[0] /= self.road_length # X-coordinate (lunghezza strada)
        pos[1] /= (self.road_width / 2) # Y-coordinate (metà larghezza strada per ottenere range [-1, 1])
        pos[2] /= 1.0 # Z-coordinate (altezza, se rilevante per la normalizzazione)


        #=====Gestione ribaltamento IMU (Roll, Pitch, Yaw)=====
        # Normalizzazione degli angoli in radianti
        # Roll e Pitch vanno da -PI a PI
        max_angle_roll_pitch = np.pi # Max angolo possibile per normalizzazione a [-1, 1]
        max_angle_yaw = np.pi # Yaw va da -PI a PI

        orientation = np.array(self.imu.getRollPitchYaw() if self.imu else [0.0, 0.0, 0.0], dtype=np.float32)

        orientation[0] /= max_angle_roll_pitch # Roll
        orientation[1] /= max_angle_roll_pitch # Pitch
        orientation[2] /= max_angle_yaw       # Yaw

        
        #=====Gestione lidars=====
        # I valori sono già in range [0, MaxRange]. La normalizzazione è corretta.
        lidar_front_values = np.array(self.lidar_front.getRangeImage(), dtype=np.float32)
        lidar_front_values[np.isinf(lidar_front_values)] = self.lidar_front.getMaxRange()

        num_samples = 10
        step = max(1, len(lidar_front_values) // num_samples)
        lidar_front_samples = lidar_front_values[::step][:num_samples]
        lidar_front_samples = lidar_front_samples / self.lidar_front.getMaxRange() #normalization

        lidar_rear_values = np.array(self.lidar_rear.getRangeImage(), dtype=np.float32)
        lidar_rear_values[np.isinf(lidar_rear_values)] = self.lidar_rear.getMaxRange()

        lidar_rear_samples = lidar_rear_values[::step][:num_samples]
        lidar_rear_samples = lidar_rear_samples / self.lidar_rear.getMaxRange() #Normalization
        #==========================================================

        #======Gestione posizione del target======
        target_coords_raw = np.array(self.target['translation'].getSFVec3f(), dtype=np.float32)
        target_coords_normalized = np.copy(target_coords_raw)
        target_coords_normalized[0] /= self.road_length 
        target_coords_normalized[1] /= (self.road_width / 2)
        target_coords_normalized[2] /= 1.0 

        # Concatenazione delle osservazioni
        obs_space = np.concatenate(
            [
            left_velocity, right_velocity, # 2 valori
            pos,                            # 3 valori (GPS x, y, z)
            orientation,                    # 3 valori (IMU Roll, Pitch, Yaw)
            lidar_front_samples,            # 10 valori
            lidar_rear_samples,              # 10 valori
            target_coords_normalized        #3 valori
            ], dtype=np.float32)
        return obs_space

    def _compute_reward(self, obs):
        # === Parse observation ===
        left_v = obs[0]
        right_v = obs[1]
        pos_denormalized = obs[2:5]
        roll, pitch, yaw = obs[5], obs[6], obs[7]
        lidar_front_samples = obs[8:18]
        lidar_rear_samples = obs[18:28]

        # === Distanza attuale dal target ===
        target_coords = self.target['translation'].getSFVec3f()
        
        # Denormalizzazione delle posizioni per il calcolo della distanza
        current_car_x = pos_denormalized[0] * self.road_length
        current_car_y = pos_denormalized[1] * (self.road_width / 2) # Ricorda la normalizzazione per metà larghezza
        current_car_z = pos_denormalized[2] * 1.0 # Altezza

        current_distance = np.linalg.norm(np.array([
            target_coords[0] - current_car_x,
            target_coords[1] - current_car_y,
            target_coords[2] - current_car_z
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
        normalized_th_front = self.collision_th / self.lidar_front.getMaxRange()
        # Modifica Cruciale: Aumento dei coefficienti delle penalità per collisione/ribaltamento
        front_collision_penalty = -1.0 if np.any(lidar_front_samples < normalized_th_front) else 0.0

        normalized_th_rear = self.collision_th/self.lidar_rear.getMaxRange()
        rear_collision_penalty = -1.0 if np.any(lidar_rear_samples < normalized_th_rear) else 0.0


        # === Penalità ribaltamento ===
        max_angle_for_fall = 0.2 # 0.2 radianti, circa 11.4 gradi
        fall_penalty = -1.0 if abs(roll * np.pi) > max_angle_for_fall or abs(pitch * np.pi) > max_angle_for_fall else 0.0


        # === Penalità sterzata brusca (penalizza angoli sterzata grandi, ma non troppo) ===
        steer_left = self.front_left_steer.getTargetPosition()
        steer_right = self.front_right_steer.getTargetPosition()
        avg_steer = 0.5 * (steer_left + steer_right)
        steer_penalty = -0.05 * (avg_steer ** 2)  # penalità quadratica più dolce

        # === Penalità per velocità differenziale fra ruote (zig-zag) ===
        velocity_penalty = -0.1 * abs(left_v - right_v)

        #===Penalità vicinanza a ostacoli frontali (media distanza lidar frontale)===
        front_lidar_penalty = 0.0
        mean_front_lidar = np.mean(lidar_front_samples)
        if mean_front_lidar < normalized_th_front * 3: # 3 volte la soglia di collisione
            front_lidar_penalty = -0.5 * (normalized_th_front * 3 - mean_front_lidar) / (normalized_th_front * 3)

        rear_lidar_penalty = 0.0
        mean_rear_lidar = np.mean(lidar_rear_samples)
        if mean_rear_lidar < normalized_th_rear * 3:
            rear_lidar_penalty = -0.5 * (normalized_th_rear * 3 - mean_rear_lidar) / (normalized_th_rear * 3)

        # === Penalità tempo (episodi lunghi) ===
        time_penalty = -0.0005 * max(0, self.curr_timestep - 1500)

        # === Bonus finale raggiungimento target ===
        target_bonus = 10.0 if current_distance < self.distance_target_threshold else 0.0

        # === Reward totale ===
        reward = (
            5.0 * progress_reward
            + 5.0 * proximity_reward
            + front_collision_penalty * 500.0 # AUMENTATO! Prima era 5.0
            + rear_collision_penalty * 500.0  # AUMENTATO! Prima era 5.0
            + fall_penalty * 500.0            # AUMENTATO! Prima era 5.0
            + steer_penalty
            + velocity_penalty
            + front_lidar_penalty * 25.0     
            + rear_lidar_penalty * 25.0      
            + target_bonus * 250.0           
            + time_penalty
        )

        return np.clip(reward, -750.0, 500.0) # Estensione del clipping per ricompense più grandi per accomodare le nuove penalità

    def _check_done(self, obs):

        cause = None

        # Controllo urto: almeno uno dei valori del lidar deve essere sotto soglia
        front_lidar_dist = obs[8:18] 
        rear_lidar_dist = obs[18:28] 

        normalized_th_front = self.collision_th / self.lidar_front.getMaxRange()
        normalized_th_rear = self.collision_th / self.lidar_rear.getMaxRange()

        collision = np.any(front_lidar_dist < normalized_th_front) or np.any(rear_lidar_dist < normalized_th_rear)
        if collision:
            cause = 'collision'

        # Controllo timeout
        timeout = self.curr_timestep >= self.max_timesteps
        if timeout and cause is None:
            cause = 'timeout'

        # controllo target
        tesla_x = obs[2] * self.road_length 
        tesla_y = obs[3] * (self.road_width / 2) 
        tesla_z = obs[4] * 1.0 

        target_coords = self.target['translation'].getSFVec3f()

        target_distance = np.sqrt(
            (target_coords[0] - tesla_x)**2 +
            (target_coords[1] - tesla_y)**2 +
            (target_coords[2] - tesla_z)**2
        )
        if target_distance < self.distance_target_threshold and cause is None:
            cause = 'target_reached'

        # controllo caduta(ribaltamento)
        roll, pitch = obs[5], obs[6] 
        max_angle_for_fall = 0.2 
        falling = abs(roll * np.pi) > max_angle_for_fall or abs(pitch * np.pi) > max_angle_for_fall
        if falling and cause is None:
            cause = 'falling'

        # === Anti-block system ===
        curr_pos = np.array([tesla_x, tesla_y, tesla_z])   
        avg_speed = 0.5 * ((90 * obs[0] + 40) + (90 * obs[1] + 40))   

        if self.last_pos is not None:
            delta_movement = np.linalg.norm(curr_pos - self.last_pos)

            if delta_movement < self.block_movement_threshold or abs(avg_speed) < self.min_speed_threshold:
                self.block_counter += 1
            else:
                self.block_counter = 0
        self.last_pos = curr_pos

        is_blocked = self.block_counter > self.max_block_steps
        if is_blocked and cause is None:
            cause = 'blocked'

        done = collision or timeout or (target_distance < self.distance_target_threshold) or falling or is_blocked

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

        # Esegui la UDR basata sul curriculum solo se non è già stata chiamata per l'episodio
        if not self.udr_called:
            self.udr_curriculum()
            self.udr_called = True

        self.car_node.setVelocity([0, 0, 0, 0, 0, 0]) #reset fisico totale della macchina
        self.car_node.resetPhysics()

        for _ in range(20): # Piccola pausa per la stabilizzazione del simulatore
            self.robot.step(self.timestep)

        # AGGIUNTA: Stampa il numero di successi consecutivi all'inizio di ogni episodio
        print(f"Inizio Episodio {self.curr_episode} | Successi consecutivi: {self.consecutive_successes}/{self.num_successes_to_advance}")


        return self._get_obs()

    def assign_objects_and_target(self, min_distance=3.0, num_obj=6, target_x_range=None):
        # Definizione dei rettilinei: start, end, coordinata costante, asse costante
        # Metti i rettilinei in ordine crescente di X per facilitare la logica
        straight_sections = [
            {'start': 15, 'end': 120, 'const': 0, 'axis': 'y'},     # rettilineo 1
            {'start': 180, 'end': 260, 'const': -20, 'axis': 'y'}   # rettilineo 2
        ]

        # Garantisci che ci siano abbastanza oggetti nel mondo per quelli da usare nella fase attuale
        if self.num_total_obstacles < num_obj:
            print(f"ATTENZIONE: Non ci sono abbastanza oggetti 'ostacolo_i' nel mondo Webots per la fase corrente del curriculum ({num_obj} richiesti, {self.num_total_obstacles} trovati). Usando tutti gli oggetti disponibili.")
            num_obj = self.num_total_obstacles

        # NUOVO: Resetta la posizione di tutti gli ostacoli (nascondili o mettili fuori scena)
        # Quelli selezionati verranno riposizionati dopo.
        # Per quelli NON selezionati, assicurati che non siano visibili.
        # Posizione di "default" fuori scena (es. molto lontano lungo l'asse X o Y)
        out_of_bounds_pos = [1000.0, 1000.0, 0.0] 
        for obstacle_data in self.random_objects:
            node = obstacle_data['node']
            if node:
                # Sposta l'ostacolo fuori dall'area di gioco visibile
                node.getField('translation').setSFVec3f(out_of_bounds_pos)
                # Resetta la rotazione per consistenza (opzionale ma buona pratica)
                node.getField('rotation').setSFRotation([0, 0, 1, 0])


        random.shuffle(self.random_objects)
        selected_obstacles = self.random_objects[:num_obj] 

        # 1. Posiziona il target in una sezione valida
        # Trova le sezioni che contengono almeno una parte del target_x_range
        valid_sections_for_target = []
        if target_x_range:
            for i, section in enumerate(straight_sections):
                # Una sezione è valida se c'è sovrapposizione tra section_x_range e target_x_range
                # Ovvero: section.end > target.start AND section.start < target.end
                if section['end'] > target_x_range[0] and section['start'] < target_x_range[1]:
                    valid_sections_for_target.append(section)
        
        if not valid_sections_for_target:
            print(f"ERRORE CRITICO: Nessuna sezione diritta valida trovata per il target_x_range: {target_x_range}")
            target_section = straight_sections[0] 
        else:
            target_section = random.choice(valid_sections_for_target)

        
        # Garantiamo che la X del target sia nell'intersezione del target_x_range e della sezione
        target_x_min = max(target_x_range[0], target_section['start'])
        target_x_max = min(target_x_range[1], target_section['end'])
        
        if target_x_min >= target_x_max:
             print(f"AVVISO: Range X target non valido per sezione {target_section}: min={target_x_min}, max={target_x_max}. Ajusting to mid-point.")
             target_x = (target_section['start'] + target_section['end']) / 2
        else:
             target_x = random.uniform(target_x_min, target_x_max)

        # Calcola l'offset laterale per il target
        target_const_offset = random.uniform((-self.road_width / 2) + 0.75, (self.road_width / 2) - 0.75)
        
        # Costruisci la posizione del target
        z_value_target = 0.03 # Z per target
        if target_section['axis'] == 'x':
            target_pos = [target_section['const'] + target_const_offset, target_x, z_value_target]
        else: # target_section['axis'] == 'y'
            target_pos = [target_x, target_section['const'] + target_const_offset, z_value_target]
        
        self.target['translation'].setSFVec3f(target_pos)
        self.target['rotation'].setSFRotation([0, 0, 1, 0])


        # 2. Posiziona gli ostacoli selezionati nelle sezioni
        all_elements_to_place_obstacles = list(selected_obstacles) 
        random.shuffle(all_elements_to_place_obstacles) 

        # Distribuisce gli ostacoli tra le sezioni della strada
        objects_per_section_obstacle = len(all_elements_to_place_obstacles) // len(straight_sections)
        remainder_obstacle = len(all_elements_to_place_obstacles) % len(straight_sections)

        current_obstacle_index = 0
        for i, section in enumerate(straight_sections):
            num_objects_in_this_section = objects_per_section_obstacle + (1 if i < remainder_obstacle else 0)
            section_objects_obstacles = all_elements_to_place_obstacles[current_obstacle_index : current_obstacle_index + num_objects_in_this_section]
            current_obstacle_index += num_objects_in_this_section

            # Inizializza con la posizione del target se l'ostacolo è nella stessa sezione del target
            placed_coords = []
            if section == target_section:
                placed_coords.append(target_x) 

            for obj in section_objects_obstacles:
                node = obj['node']
                translation_field = obj['translation']
                rotation_field = obj['rotation']

                attempts = 0
                while True:
                    # Per gli ostacoli, usa il range della sezione
                    coord = random.uniform(section['start'], section['end'])
                    
                    # Offset laterale per la larghezza della strada
                    const_offset = random.uniform((-self.road_width / 2) + 0.75, (self.road_width / 2) - 0.75)

                    # Verifica la distanza da altri oggetti già piazzati in QUESTA sezione
                    if all(abs(coord - other) >= min_distance for other in placed_coords):
                        placed_coords.append(coord)
                        break

                    attempts += 1
                    if attempts > 100:
                        print(f"AVVISO: Impossibile posizionare {node.getDef() if node else 'un ostacolo'} nel rettilineo {i+1} rispettando min_distance dopo 100 tentativi. Posizionamento casuale.")
                        break 

                # Costruisci la nuova posizione per l'ostacolo
                z_value_obstacle = 0.4 
                if section['axis'] == 'x':
                    pos = [section['const'] + const_offset, coord, z_value_obstacle]
                else: 
                    pos = [coord, section['const'] + const_offset, z_value_obstacle]
                
                translation_field.setSFVec3f(pos)
                rotation_field.setSFRotation([0, 0, 1, 0]) 

    def udr_curriculum(self):
        current_settings = self.stage_settings[self.curriculum_stage]
        target_x_range = current_settings['target_x_range']
        num_obstacles_to_use = current_settings['num_obstacles_to_use']

        #=====Posizione iniziale random della macchina=====
        rand_x_start = self.default_car_pos[0] + np.random.uniform(-3, 5) 
        rand_y_start = np.random.uniform(-0.5, 0.5) 
        self.translation_field.setSFVec3f([rand_x_start, rand_y_start, 0.4])

        #=====Rotazione iniziale random della macchina=====
        angle = np.random.uniform(-0.4, 0.4)
        self.rotation_field.setSFRotation([0, 0, 1, angle])   

        #=====Posizioniamo oggetti e target nel mondo in base alla fase del curriculum=====
        self.assign_objects_and_target(
            min_distance=5.0, 
            num_obj=num_obstacles_to_use,
            target_x_range=target_x_range 
        )
        self.target_pos = self.target_translation.getSFVec3f() 
        print(f'UDR applicata (Fase {self.curriculum_stage}): target in {self.target_pos}, {num_obstacles_to_use} ostacoli.')


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