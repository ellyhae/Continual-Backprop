import datetime
import os
import sys
import tempfile

import numpy as np
import torch as th

import wandb

from PIL import Image as PILImage
from stable_baselines3.common.logger import KVWriter, Logger, TensorBoardOutputFormat, CSVOutputFormat, HumanOutputFormat, JSONOutputFormat, Video, Image, Figure, HParam
from stable_baselines3.common.utils import get_latest_run_id

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

class WandbOutputFormat(KVWriter):
    """
    Dumps key/value pairs into Weights & Biases.
    """
    
    def __init__(self, log_dir, log_suffix):
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbOutputFormat()")
            
        wandb.define_metric("timestep")
        wandb.define_metric("*", step_metric="timestep")

    def write(self, key_values, key_excluded, step: int = 0):
        
        log_dict = {}

        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

            if excluded is not None and "wandb" in excluded:
                continue

            if isinstance(value, np.ScalarType) or isinstance(value, dict) or isinstance(value, list):
                log_dict[key] = value

            if isinstance(value, th.Tensor):
                log_dict[key] = value #  wandb.Histogram()

            if isinstance(value, Video):
                log_dict[key] = wandb.Video(value.frames, fps=value.fps)

            if isinstance(value, Figure):
                log_dict[key] = wandb.Image(value.figure)
                if value.close:
                    value.figure.close()

            if isinstance(value, Image):
                log_dict[key] = wandb.Image(PILImage.fromarray(value.image, mode="RGB"))

            if isinstance(value, HParam):
                wandb.config.update(value.hparam_dict)
                for k,v in value.metric_dict:
                    wandb.run.summary[k] = v
                    
        log_dict['timestep'] = step

        # Flush the output to the file
        wandb.log(log_dict)

    def close(self):
        """
        closes the file
        """
        wandb.finish()

def make_output_format(_format: str, log_dir: str, log_suffix: str = "", extra_loggers={}):
    """
    Function from stable_baselines3.common.logger
    Adapted to allow custom logger formats
    
    return a logger for the requested format
    :param _format: the requested format to log to ('stdout', 'log', 'json' or 'csv' or 'tensorboard')
    :param log_dir: the logging directory
    :param log_suffix: the suffix for the log file
    :return: the logger
    """
    os.makedirs(log_dir, exist_ok=True)
    if _format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif _format == "log":
        return HumanOutputFormat(os.path.join(log_dir, f"log{log_suffix}.txt"))
    elif _format == "json":
        return JSONOutputFormat(os.path.join(log_dir, f"progress{log_suffix}.json"))
    elif _format == "csv":
        return CSVOutputFormat(os.path.join(log_dir, f"progress{log_suffix}.csv"))
    elif _format == "tensorboard":
        return TensorBoardOutputFormat(log_dir)
    elif _format in extra_loggers:
        return extra_loggers[_format](log_dir, log_suffix)
    else:
        raise ValueError(f"Unknown format specified: {_format}")

def configure(folder = None, format_strings = None, extra_loggers={}):
    """
    Function from stable_baselines3.common.logger
    Adapted to allow custom logger formats
    
    Configure the current logger.
    :param folder: the save location
        (if None, $SB3_LOGDIR, if still None, tempdir/SB3-[date & time])
    :param format_strings: the output logging format
        (if None, $SB3_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    :return: The logger object.
    """
    if folder is None:
        folder = os.getenv("SB3_LOGDIR")
    if folder is None:
        folder = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    log_suffix = ""
    if format_strings is None:
        format_strings = os.getenv("SB3_LOG_FORMAT", "stdout,log,csv").split(",")

    format_strings = list(filter(None, format_strings))
    output_formats = [make_output_format(f, folder, log_suffix, extra_loggers) for f in format_strings]

    logger = Logger(folder=folder, output_formats=output_formats)
    # Only print when some files will be saved
    if len(format_strings) > 0 and format_strings != ["stdout"]:
        logger.log(f"Logging to {folder}")
    return logger

def configure_logger(
    verbose: int = 0,
    tensorboard_log = None,
    tb_log_name: str = "",
    reset_num_timesteps: bool = True,
    format_strings = [],
    extra_loggers = {},
) -> Logger:
    """
    Function from stable_baselines3.common.utils
    Adapted to allow custom logger formats
    
    Configure the logger's outputs.
    :param verbose: Verbosity level: 0 for no output, 1 for the standard output to be part of the logger outputs
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param tb_log_name: tensorboard log
    :param reset_num_timesteps:  Whether the ``num_timesteps`` attribute is reset or not.
        It allows to continue a previous learning curve (``reset_num_timesteps=False``)
        or start from t=0 (``reset_num_timesteps=True``, the default).
    :return: The logger object
    """
    save_path = None

    if tensorboard_log is not None and SummaryWriter is None:
        raise ImportError("Trying to log data to tensorboard but tensorboard is not installed.")

    if tensorboard_log is not None and SummaryWriter is not None:
        latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
        if not reset_num_timesteps:
            # Continue training in the same directory
            latest_run_id -= 1
        save_path = os.path.join(tensorboard_log, f"{tb_log_name}_{latest_run_id + 1}")
        if verbose >= 1:
            format_strings = ["stdout", "tensorboard"] + format_strings
        else:
            format_strings = ["tensorboard"] + format_strings
    elif verbose == 0:
        format_strings = [""]
    elif len(format_strings) == 0:
        format_strings = ["stdout"]
    return configure(save_path, format_strings=format_strings, extra_loggers=extra_loggers)