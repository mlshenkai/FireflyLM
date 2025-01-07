# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2025/1/3 16:06
# @File: ckpt_io
# @Email: mlshenkai@163.com
import json
import os
from typing import Union, Dict, Any, Tuple

# from transformers import PreTrainedModel
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
import torch.nn as nn
import torch.optim as optim


def load_json(file_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Load file in JSON format
    """
    with open(file=file_path, mode="r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(data: Dict[str, Any], file_path: Union[str, os.PathLike]) -> None:
    """
    Save as JSON format
    """
    with open(file=file_path, mode="w", encoding="utf-8") as fp:
        json.dump(data, fp=fp, ensure_ascii=False, indent=4)


def save_checkpoints(
    save_dir: Union[str, os.PathLike],
    booster: Booster,
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler,
    epoch: int,
    step: int,
    batch_size: int,
    coordinator: DistCoordinator,
    use_lora: bool = False,
) -> None:
    """
    save model checkpoint, optimizer, LR Scheduler and intermediate running states
    :param save_dir:
    :param booster:
    :param model:
    :param optimizer:
    :param lr_scheduler
    :param epoch:
    :param step:
    :param batch_size:
    :param coordinator:
    :param use_lora:
    :return:
    """
    save_dir = os.path.join(save_dir, f"epoch-{epoch}_step-{step}")
    os.makedirs(os.path.join(save_dir, "modeling"), exist_ok=True)
    if use_lora:
        booster.save_lora_as_pretrained(model, os.path.join(save_dir, "modeling"))
    else:
        booster.save_model(model, os.path.join(save_dir, "modeling"), shard=True)

    booster.save_optimizer(
        optimizer, checkpoint=os.path.join(save_dir, "optimizer"), shard=True
    )
    booster.save_lr_scheduler(
        lr_scheduler, checkpoint=os.path.join(save_dir, "lr_scheduler")
    )

    running_states = {
        "epoch": epoch,
        "step": step,
        "sample_start_index": step * batch_size,
    }

    if coordinator.is_master():
        save_json(running_states, os.path.join(save_dir, "running_states.json"))


def load_checkpoint(
    checkpoint_dir: Union[str, os.PathLike],
    booster: Booster,
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler,
) -> Tuple[int, int, int]:
    # Update booster params states.
    booster.load_model(model=model, checkpoint=os.path.join(checkpoint_dir, "modeling"))
    booster.load_optimizer(
        optimizer=optimizer, checkpoint=os.path.join(checkpoint_dir, "optimizer")
    )
    booster.load_lr_scheduler(
        lr_scheduler=lr_scheduler,
        checkpoint=os.path.join(checkpoint_dir, "lr_scheduler"),
    )
    running_states = load_json(
        file_path=os.path.join(checkpoint_dir, "running_states.json")
    )
    return (
        running_states["epoch"],
        running_states["step"],
        running_states["sample_start_index"],
    )
