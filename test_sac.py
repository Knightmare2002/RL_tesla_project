import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import wandb

from webots_remote_env import WebotsRemoteEnv

wandb.init(
    project="RL_tesla_project",
    name="SAC-Webots-test-run00",
    sync_tensorboard=False, # Non sincronizzare tensorboard per il test
    monitor_gym=True,
    save_code=False, # Non salvare il codice per il test
    job_type="eval" # Etichetta come job di valutazione
)

# Percorsi del modello
MODEL_DIR = "C:\\Users\samue\\OneDrive\\Desktop\\RL_tesla_project\\model_dir\\model_udr_00\model.zip"

# Numero di episodi per la valutazione
N_EVAL_EPISODES = 10

# --- Carica l'ambiente ---
env = WebotsRemoteEnv()


# --- Carica il modello addestrato ---
try:
    print(f"Caricamento del modello da: {MODEL_DIR}")
    model = SAC.load(MODEL_DIR, env=env, device='cuda')
    print("Modello caricato con successo.")
except Exception as e:
    print(f"Errore durante il caricamento del modello: {e}")
    print("Assicurati che il percorso del modello sia corretto e che il modello esista.")
    env.close()
    wandb.finish()
    exit()

# --- Valutazione del modello ---
print(f"\nInizio della valutazione del modello per {N_EVAL_EPISODES} episodi...")
try:
    # evaluate_policy è una funzione comoda di Stable Baselines3
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES, render=True) # Imposta render=True se vuoi visualizzare la simulazione
    
    print(f"\n--- Risultati della Valutazione ---")
    print(f"Reward media: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Per ottenere la lunghezza media degli episodi, dobbiamo fare un loop manuale
    # o estendere evaluate_policy (che per default non restituisce len).
    # Per semplicità, faremo un piccolo loop qui per la lunghezza media.
    episode_lengths = []
    print("\nRaccolta dati sulla lunghezza degli episodi...")
    for i in range(N_EVAL_EPISODES):
        obs, info = env.reset() # env.reset() potrebbe restituire anche 'info'
        done = False
        episode_len = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True) # deterministic=True per testare la policy finale
            obs, reward, done, _, info = env.step(action) # Aggiornato per Gymnasium (obs, reward, terminated, truncated, info)
            episode_len += 1
            if done:
                break
        episode_lengths.append(episode_len)
        print(f"Episodio {i+1}: Lunghezza {episode_len}")

    mean_episode_length = np.mean(episode_lengths)
    std_episode_length = np.std(episode_lengths)
    print(f"Lunghezza media episodi: {mean_episode_length:.2f} +/- {std_episode_length:.2f}")

    # Registra i risultati su WandB
    wandb.log({
        "test/mean_reward": mean_reward,
        "test/std_reward": std_reward,
        "test/mean_episode_length": mean_episode_length,
        "test/std_episode_length": std_episode_length
    })

except Exception as e:
    print(f"Errore durante la valutazione: {e}")
    import traceback
    traceback.print_exc() # Stampa la traceback completa per debugging


finally:
    # Chiudi l'ambiente Webots e termina la sessione WandB
    env.close()
    wandb.finish()
    print("\nTest completato.")