# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2025/1/3 19:47
# @File: misc
# @Email: mlshenkai@163.com
import os
import logging
from colossalai.cluster.dist_coordinator import DistCoordinator

# DistCoordinator().is_master()


def create_logger(coordinator: DistCoordinator, logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if coordinator.is_master():  # real logger
        additional_args = dict()
        if logging_dir is not None:
            additional_args["handlers"] = [
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ]
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            **additional_args,
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def create_tensorboard_writer(exp_dir):
    from torch.utils.tensorboard import SummaryWriter

    tensorboard_dir = f"{exp_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    return writer
