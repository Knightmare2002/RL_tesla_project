from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from webots_remote_env import WebotsRemoteEnv  # salva la classe sopra in questo file

env = WebotsRemoteEnv()

try:
    model = SAC(MlpPolicy, env, verbose=0, device='cuda')
    print(f'Using device: {model.device}')
    model.learn(total_timesteps=1_000_000, progress_bar=True)
    model.save("webots_sac_model")
finally:
    env.close()  # Assicura che venga sempre chiamato anche se il training fallisce
