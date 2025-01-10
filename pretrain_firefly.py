# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2025/1/2 21:11
# @File: pretrain_firefly
# @Email: mlshenkai@163.com
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import json
import os
import resource
from contextlib import nullcontext
import torch
from fireflyllm.datasets.loader import (
    DataCollatorForSupervisedDataset,
    StatefulDistributedSampler,
    load_tokenized_dataset,
)
from fireflyllm.datasets.dummy_dataset import RandomDataset
from fireflyllm.utils.ckpt_io import load_checkpoint, save_checkpoints
from fireflyllm.utils.config_utils import (
    define_experiment_workspace,
    save_training_config,
)
from fireflyllm.utils.froze import freeze_non_embeds_parameters
import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin
from colossalai.lazy import LazyInitContext
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from fireflyllm.utils.misc import create_logger, create_tensorboard_writer
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from fireflyllm.models.firefly_model import FireflyForCausalLM
from fireflyllm.utils.train_utils import create_colossalai_plugin
from peft import LoraConfig
# from fireflyllm.models.firefly_model import FireflyForCausalLM
# from fireflyllm.utils.ckpt_io import load_checkpoint
from fireflyllm.utils.utils import get_model_numel, format_numel_str, all_reduce_mean


def train(args: argparse.Namespace) -> None:
    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch()
    accelerator = get_accelerator()
    coordinator = DistCoordinator()

    # ==============================
    # Initialize Wandb Tensorboard and Save Config
    # ==============================
    exp_name, exp_dir = define_experiment_workspace(args)
    # coordinator.block_all()  # 等待
    if coordinator.is_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(args.__dict__, exp_dir)
        tensorboard_writer = create_tensorboard_writer(exp_dir)
        logger = create_logger(coordinator, logging_dir=exp_dir)
        if args.wandb:
            os.makedirs(os.path.join(exp_dir, "wandb"), exist_ok=True)
            wandb.init(
                project="firefly",
                name=exp_name,
                config=args.__dict__,
                dir=os.path.join(exp_dir, "wandb"),
            )

    # ==============================
    # Initialize Booster
    # ==============================
    plugin = create_colossalai_plugin(
        plugin_name=args.plugin,
        dtype=args.mixed_precision,
        grad_clip=args.grad_clip,
        use_flash_attn=args.use_flash_attn,
        use_grad_checkpoint=args.use_grad_checkpoint,
        accumulation_steps=args.accumulation_steps,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        sp_size=args.sp_size,
        sp_mode=args.sp_mode,
        zero_stage=args.zero_stage,
        enable_sequence_parallelism=args.enable_sequence_parallelism,
        zero_cpu_offload=args.zero_cpu_offload,
        microbatch_size=args.microbatch_size,
    )

    booster = Booster(plugin=plugin)

    # ======================================================
    # Initialize Tokenizer, Dataset, Collator and Dataloader
    # ======================================================
    tokenizer = AutoTokenizer.from_pretrained(args.token_pretrained_path)
    coordinator.print_on_master("load dataset...")
    if args.benchmark:
        coordinator.print_on_master(
            f"Run benchmark with {args.num_samples} random samples."
        )
        dataset = RandomDataset(
            num_samples=args.num_samples,
            max_length=args.max_length,
            vocab_size=args.vocab_size,
        )
        dataloader = plugin.prepare_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            seed=42,
            drop_last=True,
            distributed_sampler_cls=StatefulDistributedSampler,
        )

    else:
        dataset = load_tokenized_dataset(args.dataset, mode="train")
        data_collator = DataCollatorForSupervisedDataset(
            tokenizer=tokenizer, max_length=args.max_length, padding=args.padding
        )
        dataloader = plugin.prepare_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            seed=42,
            drop_last=True,
            collate_fn=data_collator,
            distributed_sampler_cls=StatefulDistributedSampler,
        )

    coordinator.print_on_master(
        f"Max device memory after data loader: {accelerator.max_memory_allocated() / 1024 ** 2:.2f} MB"
    )

    # ======================================================
    # Initialize Model, Objective, Optimizer and LR Scheduler
    # ======================================================
    # When training the ChatGLM model, LoRA and gradient checkpointing are incompatible.

    init_ctx = (
        LazyInitContext(default_device=get_current_device())
        if isinstance(plugin, (GeminiPlugin, HybridParallelPlugin))
        and args.lora_rank == 0
        else nullcontext()
    )
    with init_ctx:
        # model = AutoModelForCausalLM.from_pretrained(
        #     args.pretrained,
        #     torch_dtype=(
        #         torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
        #     ),
        #     trust_remote_code=True,
        # )
        config = AutoConfig.from_pretrained(args.pretrained, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16,
            trust_remote_code=True)
        # model = model.to(torch.bfloat16)
        if args.freeze_non_embeds_params:
            freeze_non_embeds_parameters(model)

    # 使用lora, 当开启lora时最好将 freeze_non_embeds_params 设置为True
    if args.lora_rank > 0:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=32, lora_dropout=0.1
        )
        model = booster.enable_lora(model, lora_config=lora_config)

    # this is essential, otherwise the grad checkpoint will not work.
    model.train()

    if args.use_grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        coordinator.print_on_master(msg="Gradient checkpointing enabled successfully")

    model_numel = get_model_numel(model)
    coordinator.print_on_master(f"Model params: {format_numel_str(model_numel)}")

    optimizer = HybridAdam(
        model_params=(
            filter(lambda p: p.requires_grad, model.parameters())
            if args.freeze_non_embeds_params
            else model.parameters()
        ),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        adamw_mode=True,
    )  # 混合精度训练 选择Hybrid下面的优化器还是很不错的

    if args.warmup_steps is None:  # 必须要有warmup 否则在训练的过程中抖到你怀疑人生
        args.warmup_steps = (
            args.num_epochs
            * 0.0025
            * (
                len(dataloader) // args.accumulation_steps
            )  # accumulation_steps 表示梯度累计的步数，
            # len(dataloader) // args.accumulation_steps 表示一个batch中梯度更新的次数
        )  # warmup的步数 使用整个训练过程中梯度更新次数的千分之2.5
        coordinator.print_on_master(rf"Warmup steps is set to {args.warmup_steps}")

    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer,
        total_steps=args.num_epochs * (len(dataloader) // args.accumulation_steps),
        warmup_steps=args.warmup_steps,
        eta_min=0.1 * args.lr if args.min_lr is None else args.min_lr,
    )

    # ======================================================
    # Enhance model optimizer, dataloader lr_scheduler
    # ======================================================
    # default_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    # torch.set_default_dtype(default_dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        lr_scheduler=lr_scheduler,
    )
    # torch.set_default_dtype(torch.float)
    coordinator.print_on_master(
        f"Booster init max device memory: {accelerator.max_memory_allocated() / 1024 ** 2:.2f} MB"
    )
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB"
    )

    start_epoch = 0
    start_step = 0
    sample_start_idx = 0
    if args.load_checkpoint is not None:
        if "modeling" in args.load_checkpoint:
            coordinator.print_on_master(
                f"Continued pretrain from checkpoint {args.load_checkpoint}"
            )
            booster.load_model(model, args.load_checkpoint)

        else:
            coordinator.print_on_master(
                f"Load model checkpoint from {args.load_checkpoint}"
            )
            start_epoch, start_step, sample_start_idx = load_checkpoint(
                args.load_checkpoint, booster, model, optimizer, lr_scheduler
            )
            coordinator.print_on_master(
                f"Load checkpoint {args.load_checkpoint} at epoch {start_epoch} step {start_step}"
            )
            coordinator.print_on_master(f"Loaded sample at index {sample_start_idx}")

        coordinator.print_on_master(
            f"Checkpoint loaded device memory {accelerator.max_memory_allocated() / 1024 ** 2:0.2f}MB"
        )
        coordinator.print_on_master(
            f"Checkpoint loaded device memory {accelerator.memory_allocated() / 1024 ** 2:.2f}MB"
        )
        coordinator.print_on_master(
            f"Checkpoint loaded max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB"
        )

    num_steps_per_epoch = len(dataloader) // args.accumulation_steps

    # 如果是继续训练 则设置sample的start index为 sampler_start_index
    assert isinstance(dataloader.sampler, StatefulDistributedSampler)
    dataloader.sampler.set_start_index(start_index=sample_start_idx)

    for epoch in range(start_epoch, args.num_epochs):
        dataloader.sampler.set_epoch(epoch)
        if isinstance(plugin, HybridParallelPlugin) and plugin.pp_size > 1:
            data_iter = iter(dataloader)
            step_bar = tqdm(
                range(len(dataloader)),
                desc="Step",
                disable=not (
                    coordinator.local_rank == coordinator.world_size - 1
                ),  # 非master 不展示
            )
            for step in step_bar:
                outputs = booster.execute_pipeline(
                    data_iter,
                    model,
                    criterion=lambda outputs, inputs: outputs[0],
                    optimizer=optimizer,
                    return_loss=True,
                )
                loss = outputs["loss"]

                if (
                    booster.plugin.stage_manager.is_last_stage
                ):  # 最后一个阶段执行all_reduce_mean 计算全局梯度
                    global_loss = all_reduce_mean(loss, plugin)
                    if coordinator.local_rank == coordinator.world_size - 1:
                        step_bar.set_postfix({"train/loss": global_loss.item()})

                optimizer.step()
                optimizer.zero_grad()

                # Save model
                save_model_condition = (
                    args.save_interval > 0 and (step + 1) % args.save_interval == 0
                )

                if not args.skip_save_each_epoch:
                    save_model_condition = save_model_condition or (step + 1) % len(
                        dataloader
                    )

                if save_model_condition and not args.benchmark:
                    coordinator.print_on_master(
                        "\n Start saving model checkpoint with running states"
                    )

                    accelerator.empty_cache()
                    save_checkpoints(
                        save_dir=args.save_dir,
                        booster=booster,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        step=step + 1,
                        batch_size=args.batch_size,
                        coordinator=coordinator,
                        use_lora=(args.lora_rank > 0),
                    )
                    coordinator.print_on_master(
                        f"Saved checkpoint at epoch {epoch} step {step+1} at folder {args.save_dir}"
                    )

        else:
            pbar = tqdm(
                desc=f"Epoch {epoch}",
                disable=not coordinator.is_master(),
                total=num_steps_per_epoch,
                initial=start_step // args.accumulation_steps,
            )
            total_loss = torch.tensor(0.0, device=get_current_device())
            for step, batch in enumerate(dataloader, start=start_step):
                batch = {k: v.to(get_current_device()) for k, v in batch.items()}

                batch_output = model(**batch)

                loss = batch_output.loss / args.accumulation_steps  # 计算当前loss

                total_loss.add_(loss.data)  # 将当前loss 添加到总loss中方便计算累计梯度

                booster.backward(loss, optimizer)  # 计算当前梯度

                if (step + 1) % args.accumulation_steps == 0:  # 到达累计步数
                    # 执行反向传播
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    all_reduce_mean(tensor=total_loss)
                    pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})
                    if coordinator.is_master():
                        global_step = (epoch * num_steps_per_epoch) + (
                            step + 1
                        ) // args.accumulation_steps  # 当前的步数
                        tensorboard_writer.add_scalar(
                            tag="Loss",
                            scalar_value=total_loss.item(),
                            global_step=global_step,
                        )
                        tensorboard_writer.add_scalar(
                            tag="Learning Rate",
                            scalar_value=lr_scheduler.get_last_lr()[0],
                            global_step=global_step,
                        )
                    total_loss.fill_(0.0)
                    pbar.update()

                # Save model
                save_model_condition = (
                    args.save_interval > 0 and (step + 1) % args.save_interval == 0
                )

                if not args.skip_save_each_epoch:
                    save_model_condition = save_model_condition or (step + 1) % len(
                        dataloader
                    )

                if save_model_condition and not args.benchmark:
                    coordinator.print_on_master(
                        "\n Start saving model checkpoint with running states"
                    )

                    accelerator.empty_cache()
                    save_checkpoints(
                        save_dir=args.save_dir,
                        booster=booster,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        step=step + 1,
                        batch_size=args.batch_size,
                        coordinator=coordinator,
                        use_lora=(args.lora_rank > 0),
                    )
        # Delete cache.
        # del batch, batch_labels, batch_output, loss
        accelerator.empty_cache()

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(start_index=0)
        start_step = 0

    # Final save.
    if not args.benchmark:
        coordinator.print_on_master("Start saving final model checkpoint")
        booster.save_model(model, os.path.join(args.save_dir, "modeling"), shard=True)
        coordinator.print_on_master(
            f"Saved final model checkpoint at epoch {epoch} at folder {args.save_dir}"
        )

    coordinator.print_on_master(
        f"Max device memory usage: {accelerator.max_memory_allocated()/1024**2:.2f} MB"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Basic training information.
    parser.add_argument(
        "--pretrained",
        type=str,
        default="/code-online/code/FireflyLM/fireflyllm/models",
        help="Address of the pre-trained model",
    )
    parser.add_argument(
        "--token_pretrained_path",
        type=str,
        default="/code-online/code/FireflyLM/fireflyllm/qwen_tokenizer",

    )

    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Load checkpoint for continuous training.",
    )
    parser.add_argument("--dataset", nargs="+", default=[])
    parser.add_argument(
        "--plugin",
        type=str,
        default="gemini",
        choices=["gemini", "gemini_auto", "zero2", "zero2_cpu", "3d", "ddp"],
        help="Choose which plugin to use",
    )
    parser.add_argument("--save_interval", type=int, default=1000, help="Save interval")
    parser.add_argument(
        "--save_dir", type=str, default="checkpoint_dir", help="Checkpoint directory"
    )
    parser.add_argument(
        "--outputs", type=str, default="outputs"
    )
    parser.add_argument(
        "--tensorboard_dir", type=str, default="logs_dir", help="Tensorboard directory"
    )
    parser.add_argument(
        "--config_file", type=str, default="config_file", help="Config file"
    )
    # Training parameters
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="Number of accumulation steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Global Batch size of each process"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Model max length")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16"],
        help="Mixed precision",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument(
        "--use_grad_checkpoint",
        action="store_true",
        default=False,
        help="Use gradient checkpointing",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        default=True,
        help="Use flash-attention",
    )
    parser.add_argument(
        "--use_neft",
        action="store_true",
        default=False,
        help="Use NEFTune",
    )
    parser.add_argument(
        "--freeze_non_embeds_params",
        action="store_true",
        default=False,
        help="Freeze non embeddings parameters",
    )
    parser.add_argument("--pad_token", choices=["eos", "unk"], default="eos")
    parser.add_argument(
        "--padding_mode", choices=["max_length", "longest"], default="max_length"
    )
    parser.add_argument(
        "--skip_save_each_epoch",
        action="store_true",
        default=False,
        help="Skip saving the model checkpoint after each epoch is completed.",
    )

    # Additional arguments for 3d plugin.
    parser.add_argument(
        "--tp_size", type=int, default=1, help="TP size, used for 3d plugin."
    )
    parser.add_argument(
        "--pp_size", type=int, default=1, help="PP size, used for 3d plugin."
    )
    parser.add_argument(
        "--sp_size", type=int, default=1, help="SP size, used for 3d plugin."
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="Zero stage, used for 3d plugin.",
        choices=[0, 1, 2],
    )
    parser.add_argument(
        "--sp_mode",
        type=str,
        default="split_gather",
        choices=["split_gather", "ring", "all_to_all"],
        help="SP mode, used for 3d plugin.",
    )
    parser.add_argument(
        "--enable_sequence_parallelism",
        default=False,
        action="store_true",
        help="Whether to enable SP, used for 3d plugin.",
    )
    parser.add_argument(
        "--zero_cpu_offload",
        default=False,
        action="store_true",
        help="Whether to use offloading, used for 3d plugin.",
    )
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=1,
        help="Batch size for each process in PP, used for 3d plugin.",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=0, help="lora rank when using lora to train."
    )

    # Additional arguments for benchmark.
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples for benchmarking.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="Benchmark performance using random dataset.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=151936
    )
    args = parser.parse_args()
    train(args)
