from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
    
    def on_rollout_end(self):
        self.logger.record("train/reward", self.locals['self'].rollout_buffer.rewards.sum())
        
    def _on_step(self):
        return True