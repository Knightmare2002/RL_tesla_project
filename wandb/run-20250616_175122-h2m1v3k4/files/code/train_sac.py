import os
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb

from webots_remote_env import WebotsRemoteEnv

wandb.init(
    project="RL_tesla_project",          
    name="SAC-Webots-run",             
    sync_tensorboard=True,             
    monitor_gym=True,                  
    save_code=True
)

# Crea ambiente Webots
env = WebotsRemoteEnv()

# Percorsi per salvataggio
CHECKPOINT_DIR = "C:\\Users\\samue\\OneDrive\\Desktop\\MLDL\\RL_tesla_project\\checkpoint_dir"
MODEL_DIR = "C:\\Users\\samue\\OneDrive\\Desktop\\MLDL\\RL_tesla_project\\model_dir"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Callback per salvataggi periodici
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,                     # salva ogni 100k timesteps
    save_path=CHECKPOINT_DIR,
    name_prefix="sac_model"
)

# Callback per integrazione con wandb
wandb_callback = WandbCallback(
    gradient_save_freq=0,
    model_save_path=MODEL_DIR,
    verbose=2,
)

try:
    model = SAC(MlpPolicy, env, verbose=1, device='cuda', tensorboard_log="./tb_logs/")
    print(f'Using device: {model.device}')

    model.learn(
        total_timesteps=1_000_000,
        progress_bar=True,
        callback=[checkpoint_callback, wandb_callback]
    )

    # Salva modello finale
    model.save(os.path.join(MODEL_DIR, "webots_sac_final"))

finally:
    env.close()
    wandb.finish()
