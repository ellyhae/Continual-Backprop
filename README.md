# Continual-Backprop
Implementations and experiments for Continual Backprop (CBP) and Continual PPO (CPPO).

Currently:
- Adam implementation that allows for resetting individual weights' 'step' counter
- Jupyter notebook for trial and error experiments
- Pre-calculated 1M steps of bitflip environment data for easier validation of changed algorithms
- SlipperyAnt environment based on Pybullets' Ant environment. Periodically changes the friction between ant and ground
- CBP implementation. There are still open questions about handling batches and replacement rate
- CPPO based on PPO implementation by stable baselines 3
- Training notebook for everything related to training CPPO
- Tensorboard logging.

## WandB

On Windows: remember to activate Developer Mode, otherwise WandB cannot sync Tensorboard data (relies on symlinks that are not available otherwise)

## Tensorboard

```console
tensorboard --logdir [path, e.g. hydra_outputs/date/time]
```

## Hydra

```console
python hydra_launcher.py --multirun algorithm=ppo,cppo1 random=seed1,seed2
```

or

```console
python hydra_launcher.py --multirun algorithm=glob(*) random=glob(*)
```