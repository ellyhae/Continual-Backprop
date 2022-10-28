import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import product

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import PPO

from cppo import CPPO_Policy
from env import SlidingAntEnv

from stable_baselines3.common.monitor import Monitor

class WeightLogger(BaseCallback):
    def _on_step(self):
        return True
    
    def _on_rollout_start(self):
        self._log()
        
    def _on_training_end(self):
        self._log()
        
    def _log(self):
        params = np.abs(self.model.policy.parameters_to_vector())
        self.logger.record("train/weight_magnitude_mean", float(params.mean()))
        self.logger.record("train/weight_magnitude_std", float(params.std()))
        self.logger.dump(self.num_timesteps)

class AgesLogger(BaseCallback):
    def __init__(self, save_dir, verbose=0):
        super(AgesLogger, self).__init__(verbose)
        self.save_dir = save_dir
        self.iteration = 0
        
    def _init_callback(self):
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _on_rollout_start(self):
        self._save()
        
    def _on_training_end(self):
        self._save()
        
    def _on_step(self):
        return True
    
    def _save(self):
        if hasattr(self.model.policy.optimizer, 'cbp_vals'):
            ages = {}
            for model, model_name in zip(self.model.policy.optimizer.linear_layers, ['policy', 'value']):
                for i, layer in enumerate(model):
                    ages[f'{model_name}_{i}'] = self.model.policy.optimizer.cbp_vals[layer]['age'].cpu().numpy()
               
            np.savez_compressed(os.path.join(self.save_dir, str(self.iteration)+'.npz'), **ages)
                
            self.iteration += 1
    
#class SlidingEval(EvalCallback):
#    def __init__(self, **kwargs):
#        super(SlidingEval, self).__init__(Monitor(SlidingAntEnv(change_steps=np.inf)), eval_freq=0, **kwargs)
#        
#    def _on_rollout_start(self):
#        self._eval()
#        
#    def _on_training_end(self):
#        self._eval()
#    
#    def _eval(self):
#        self.eval_freq = 1
#        for e in self.eval_env.envs:
#            e.env.friction = self.training_env.envs[0].env.friction
#        super(SlidingEval, self)._on_step()
#        self.eval_freq = 0

class SlidingEval(BaseCallback):
    def __init__(self, max_steps, deterministic=True, n_eval_episodes=1):
        super(SlidingEval, self).__init__()
        self.eval_env = DummyVecEnv([lambda: Monitor(SlidingAntEnv(change_steps=np.inf, max_steps=max_steps))])
        self.deterministic = deterministic
        self.n_eval_episodes = n_eval_episodes
        
    def _on_rollout_start(self):
        self._eval()
        
    def _on_training_end(self):
        self._eval()
    
    def _eval(self):
        for e in self.eval_env.envs:
            e.env.friction = self.training_env.envs[0].env.friction
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            return_episode_rewards=True
        )
        self.logger.record("eval/mean_reward", float(np.mean(episode_rewards)))
        self.logger.record("eval/mean_ep_length", np.mean(episode_lengths))
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)
        
    def _on_step(self):
        return True
    
def get_mean_tag(tag, log_dir):
    exp_dirs = pd.Series(os.listdir(log_dir))
    names = exp_dirs.str.split('_').str[0].unique()
    dat = []
    for name in names:
        runs = exp_dirs[exp_dirs.str.startswith(name)]
        vals = []
        for run in runs:
            ea = EventAccumulator(os.path.join(log_dir, run))
            ea.Reload()
            s = pd.DataFrame(ea.Scalars(tag)).set_index('step').value
            vals.append(s)
        r = pd.concat(vals, axis=1).mean(1)
        r.name = name
        dat.append(r)
    return pd.concat(dat, axis=1)

def plot_ages(ages_dir, name, iteration): # TODO add animation
    with np.load(os.path.join(ages_dir, name, str(iteration)+'.npz')) as ages:
        plt.violinplot(list(ages.values()))
        plt.xticks(range(1, len(ages.files)+1), ages.files)
    plt.show()
    
def run_experiment(name, learner_class, params, seed, total_timesteps, n_jumps, eval_args, model_dir, ages_dir):
    env = Monitor(SlidingAntEnv(total_timesteps//n_jumps))
    ppo = PPO(learner_class, env, seed=int(seed), **params)
    callbacks = [WeightLogger(), AgesLogger(os.path.join(ages_dir, name)), SlidingEval(**eval_args)]
    ppo.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name=name)
    ppo.save(os.path.join(model_dir, name))
    
def get_cppo_settings(settings, cppo_options, cppo_option_index):
    cppo_settings = deepcopy(settings)
    cppo_settings['policy_kwargs']['optimizer_kwargs'] |= cppo_options[cppo_option_index]
    return cppo_settings

def get_experiment_combinations(n_repetitions, settings, cppo_options, total_timesteps, n_jumps, eval_args, model_dir, ages_dir, entropy):
    seed_sequence = np.random.SeedSequence(entropy)
    seeds = seed_sequence.generate_state(n_repetitions)
    
    experiments = [('baseline_'+str(i), "MlpPolicy", settings, seed, total_timesteps, n_jumps, eval_args, model_dir, ages_dir) for i, seed in enumerate(seeds)]
    experiments += [(f'cppo {option}_{i}', CPPO_Policy, get_cppo_settings(settings, cppo_options, option), seeds[i], total_timesteps, n_jumps, eval_args, model_dir, ages_dir) 
                    for option, i in product(range(len(cppo_options)), range(n_repetitions))]
    
    return experiments