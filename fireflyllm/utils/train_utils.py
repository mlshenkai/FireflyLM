# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2025/1/3 18:21
# @File: get_plugin
# @Email: mlshenkai@163.com
from colossalai.booster.plugin import (
    GeminiPlugin,
    HybridParallelPlugin,
    LowLevelZeroPlugin,
    TorchDDPPlugin,
    TorchFSDPPlugin,
    Plugin,
)
import argparse
from colossalai.accelerator import get_accelerator


def create_colossalai_plugin(
    plugin_name,
    dtype,
    grad_clip,
    use_flash_attn: bool,
    use_grad_checkpoint: bool,
    accumulation_steps: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    sp_size: int = 1,
    sp_mode: str = "all_to_all",
    zero_stage: int = 0,
    enable_sequence_parallelism: bool = False,
    zero_cpu_offload: bool = False,
    microbatch_size: int = None,
) -> Plugin:
    if plugin_name == "ddp":
        plugin = TorchDDPPlugin(
            find_unused_parameters=True if use_grad_checkpoint is False else False
        )
    elif plugin_name == "gemini":
        plugin = GeminiPlugin(
            precision=dtype,
            initial_scale=2**16,
            max_norm=grad_clip,
            enable_gradient_accumulation=(accumulation_steps > 1),
            enable_fused_normalization=get_accelerator().is_available(),
            enable_flash_attention=use_flash_attn,
        )
    elif plugin_name == "gemini_auto":
        plugin = GeminiPlugin(
            precision=dtype,
            placement_policy="auto",
            initial_scale=2**16,
            max_norm=grad_clip,
            enable_gradient_accumulation=(accumulation_steps > 1),
            enable_fused_normalization=get_accelerator().is_available(),
            enable_flash_attention=use_flash_attn,
        )
    elif plugin_name == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=dtype,
            initial_scale=2**16,
            max_norm=grad_clip,
        )
    elif plugin_name == "zero2_cpu":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=dtype,
            initial_scale=2**16,
            cpu_offload=True,
            max_norm=grad_clip,
        )
    elif plugin_name == "3d":
        plugin = HybridParallelPlugin(
            tp_size=tp_size,
            pp_size=pp_size,
            sp_size=sp_size,
            sequence_parallelism_mode=sp_mode,
            zero_stage=zero_stage,
            enable_flash_attention=use_flash_attn,
            enable_fused_normalization=get_accelerator().is_available(),
            enable_sequence_parallelism=enable_sequence_parallelism,
            cpu_offload=(True if zero_stage >= 1 and zero_cpu_offload else False),
            max_norm=grad_clip,
            precision=dtype,
            microbatch_size=microbatch_size,
        )
    else:
        raise ValueError(f"Unknown plugin {plugin_name}")
    return plugin

