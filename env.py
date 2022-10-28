import numpy as np
from pybullet_envs.gym_locomotion_envs import AntBulletEnv

class SlidingAntEnv(AntBulletEnv):
    '''
    Change the friction between Ant and Floor every change_steps timesteps to a random number between 10**log_low and 10**log_high
    Limit length of episodes with max_steps. Needed for evaluation, otherwise leads to infinite episodes
    '''
    def __init__(self, change_steps, log_low=-4, log_high=4, seed=42, max_steps=None, **kwargs):
        super(SlidingAntEnv, self).__init__(**kwargs)
        self.change_steps = change_steps
        self.log_low = log_low
        self.log_high = log_high
        self.rng = np.random.default_rng(seed)
        self.counter = 0 # won't be reset when the environment is reset
        self.friction = 10**self.rng.uniform(self.log_low, self.log_high)
        self.max_steps = max_steps
        self.n_steps = 0
    
    def reset(self):
        ret = super(SlidingAntEnv, self).reset()
        self._set_friction()
        self.n_steps = 0
        return ret
        
    def step(self, a):
        observation, reward, done, info = super(SlidingAntEnv, self).step(a)
        self.counter += 1
        self.n_steps += 1
        if self.counter == self.change_steps:
            self.counter = 0
            self.friction = 10**self.rng.uniform(self.log_low, self.log_high)
            self._set_friction()
        if self.max_steps is not None and self.n_steps == self.max_steps:   # if step limit is reached, set done=True
            done = True
            self.n_steps = 0
        return observation, reward, done, info
    
    def _set_friction(self):
        for body_id, joint_id in self.ground_ids:
            self._p.changeDynamics(body_id, joint_id, lateralFriction=self.friction)
            