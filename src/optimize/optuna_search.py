from typing import Dict, Any

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def objective(trial: optuna.Trial, make_env_fn, total_timesteps: int = 100_000) -> float:
	env = DummyVecEnv([make_env_fn])
	policy_kwargs = dict(net_arch=[trial.suggest_int("layer1", 64, 256), trial.suggest_int("layer2", 32, 128)])
	gamma = trial.suggest_float("gamma", 0.90, 0.9999, log=True)
	learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
	ent_coef = trial.suggest_float("ent_coef", 1e-5, 0.05, log=True)
	clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
	n_steps = trial.suggest_int("n_steps", 128, 4096, log=True)
	batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
	n_epochs = trial.suggest_int("n_epochs", 5, 20)

	model = PPO("MlpPolicy", env, verbose=0, gamma=gamma, learning_rate=learning_rate, ent_coef=ent_coef, clip_range=clip_range, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, policy_kwargs=policy_kwargs)
	model.learn(total_timesteps=total_timesteps)

	# Evaluate average episodic return (environment designed for daily steps, so sum of rewards is cumulative return approximation)
	obs, _ = env.reset()
	done = False
	acc = 0.0
	while not done:
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, done, _, _ = env.step(action)
		acc += float(reward)
	return acc


def run_study(make_env_fn, n_trials: int = 20, total_timesteps: int = 100_000) -> optuna.Study:
	study = optuna.create_study(direction="maximize")
	study.optimize(lambda trial: objective(trial, make_env_fn, total_timesteps), n_trials=n_trials)
	return study