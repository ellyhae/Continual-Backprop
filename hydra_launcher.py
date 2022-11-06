# python hydra_launcher.py --multirun algorithm=glob(*) random=glob(*)

import os

from torch import nn

from omegaconf import DictConfig, OmegaConf
import hydra

from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor

from cppo import CPPO_Policy
from env import SlidingAntEnv

from env import SlidingAntEnv
from utils import WeightLogger, AgesLogger, SlidingEval

@hydra.main(version_base=None, config_path="hydra_config", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    env = Monitor(SlidingAntEnv(cfg.total_timesteps//cfg.n_jumps, max_steps=cfg.train_max_steps))
    learner_class = "MlpPolicy" if cfg.algorithm.policy == "MlpPolicy" else CPPO_Policy
    settings = OmegaConf.to_container(cfg.algorithm.settings)
    settings['policy_kwargs']['activation_fn'] = getattr(nn, settings['policy_kwargs']['activation_fn'])
    ppo = PPO(learner_class, env, seed=int(cfg.random.seed), **settings)
    callbacks = [WeightLogger(), AgesLogger(cfg.ages_dir), SlidingEval(**cfg.eval)]
    ppo.learn(total_timesteps=cfg.total_timesteps, callback=callbacks, tb_log_name=cfg.name)
    ppo.save(os.path.join(cfg.model_dir, cfg.name))

if __name__ == "__main__":
    run_experiment()