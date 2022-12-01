# python hydra_launcher.py --multirun algorithm=glob(*) random=glob(*)

import os

import numpy as np
import torch

import wandb
from wandb.integration.sb3 import WandbCallback

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

from cppo import CPPO_Policy
from env import SlidingAntEnv

from env import SlidingAntEnv
from utils import WeightLogger, AgesLogger, SlidingEval
from sb3_logger import configure_logger, WandbOutputFormat

@hydra.main(version_base=None, config_path="hydra_config", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    
    # just in case, set global seed
    np.random.seed(cfg.random.seed)
    torch.manual_seed(cfg.random.seed)
    
    #hydra_cfg = HydraConfig.get()
    
    run = wandb.init(
        project="CPPO",
        group=cfg.wandb.group,
        name=cfg.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=cfg.wandb.dir,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        mode=cfg.wandb.mode,
        reinit=True
    )
    
    vec_env_cls = DummyVecEnv if cfg.n_envs == 1 else SubprocVecEnv
    
    env = make_vec_env(SlidingAntEnv, cfg.n_envs, seed=cfg.random.seed, vec_env_cls=vec_env_cls, env_kwargs={'change_steps':(cfg.total_timesteps//cfg.n_jumps)//cfg.n_envs, 'max_steps':cfg.train_max_steps, 'seed':cfg.random.seed})
    
    #env = VecVideoRecorder(env, cfg.video_dir, record_video_trigger=lambda x: x % (cfg.total_timesteps // (cfg.n_jumps * 2)) == 0, video_length=500)
    
    learner_class = "MlpPolicy" if cfg.algorithm.policy == "MlpPolicy" else CPPO_Policy
    
    settings = OmegaConf.to_container(cfg.algorithm.settings, resolve=True)
    settings['policy_kwargs']['activation_fn'] = getattr(torch.nn, settings['policy_kwargs']['activation_fn'])
    settings['n_steps'] = settings['n_steps'] // cfg.n_envs
    
    ppo = PPO(learner_class, env, seed=int(cfg.random.seed), **settings)
    
    tb_log_dir = settings['tensorboard_log']
    tb_log_name = cfg.name
    ppo.set_logger(configure_logger(0, tb_log_dir, tb_log_name, True, extra_formats=[WandbOutputFormat]))
    
    callbacks = [WeightLogger(), AgesLogger(cfg.ages_dir), SlidingEval(**cfg.eval), WandbCallback(gradient_save_freq=cfg.wandb.gradient_save_freq)]
    ppo.learn(total_timesteps=cfg.total_timesteps, callback=callbacks, tb_log_name=tb_log_name)
    
    ppo.save(os.path.join(cfg.model_dir, cfg.name))
    env.close()
    ppo.logger.close()

if __name__ == "__main__":
    wandb.setup()
    run_experiment()