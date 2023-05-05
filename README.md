# Continual-Backprop
Implementations and experiments for Continual Backprop (CBP) and Continual PPO (CPPO).

Includes:
- SlipperyAnt environment based on Pybullets' Ant environment. Periodically changes the friction between ant and ground
- Adam implementation that allows for resetting individual weights' 'step' counter
- CPB+Adam implementation
- CPPO based on PPO implementation by stable baselines 3
- Weights & Biases logger for Stable baselines 3. Logs available at [here](https://wandb.ai/hae_/CPPO)
- Several logging callbacks
- Hydra configuration management
- Multiple old Jupyter notebooks. Used for development and testing, preserved for future reference

## Start experiments

Activate the conda environment and then use a command similar to

```console
python hydra_launcher.py --multirun algorithm=ppo,cppo1 random=seed1,seed2
```

or

```console
python hydra_launcher.py --multirun algorithm=glob(*) random=glob(*)
```