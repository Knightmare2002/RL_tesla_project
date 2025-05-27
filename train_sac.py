from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from env.webots_remote_env import WebotsRemoteEnv  # salva la classe sopra in questo file

env = WebotsRemoteEnv()

model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

model.save("webots_sac_model")
env.close()
