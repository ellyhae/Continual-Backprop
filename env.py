import numpy as np
from pybulletgym.envs.mujoco.envs.locomotion.ant_env import AntMuJoCoEnv

class SlindingAntEnv(AntMuJoCoEnv):
    '''
    Change the friction between Ant and Floor every change_steps timesteps to a random number between 10**log_low and 10**log_high
    '''
    def __init__(self, change_steps, log_low=-4, log_high=4):
        super(SlindingAntEnv, self).__init__()
        self.change_steps = change_steps
        self.log_low = log_low
        self.log_high = log_high
        self.counter = 0 # won't be reset when the environment is reset
        
    def step(self, a):
        ret = super(SlindingAntEnv, self).step(a)
        self.counter += 1
        if self.counter == self.change_steps:
            self.counter = 0
            self._p.changeDynamics(self.robot_body.bodies[self.robot_body.bodyIndex], -1, lateralFriction=10**np.random.uniform(self.log_low, self.log_high))
        return ret