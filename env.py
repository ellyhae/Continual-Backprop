import numpy as np
from pybullet_envs.gym_locomotion_envs import AntBulletEnv

class SlindingAntEnv(AntBulletEnv):
    '''
    Change the friction between Ant and Floor every change_steps timesteps to a random number between 10**log_low and 10**log_high
    '''
    def __init__(self, change_steps, log_low=-4, log_high=4, seed=42):
        super(SlindingAntEnv, self).__init__()
        self.change_steps = change_steps
        self.log_low = log_low
        self.log_high = log_high
        self.rng = np.random.default_rng(seed)
        self.counter = 0 # won't be reset when the environment is reset
        self._p.changeDynamics(self.robot_body.bodies[self.robot_body.bodyIndex], -1, lateralFriction=10**self.rng.uniform(self.log_low, self.log_high)) # change initial friction
        
    def step(self, a):
        ret = super(SlindingAntEnv, self).step(a)
        self.counter += 1
        if self.counter == self.change_steps:
            self.counter = 0
            self._p.changeDynamics(self.robot_body.bodies[self.robot_body.bodyIndex], -1, lateralFriction=10**self.rng.uniform(self.log_low, self.log_high))
        return ret