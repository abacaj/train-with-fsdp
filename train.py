from tqdm import tqdm
from datetime import datetime
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeBlock
from core import (
    DEFAULT_EOS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_UNK_TOKEN,
    DataCollatorForSupervisedDataset,
    SupervisedDataset,
    save_model,
    clip_model_gradients,
    disable_model_dropout,
    get_all_reduce_mean,
    get_dataloaders,
    get_optimizer,
    get_scheduler,
)
from dotenv import load_dotenv

import functools
import torch.distributed as dist
import wandb
import uuid
import torch
import transformers
import os

load_dotenv()


def setup_model(model_name, max_length):
    config = transformers.AutoConfig.from_pretrained(
        model_name,
        use_auth_token=os.environ["HF_TOKEN"],
    )

    config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=os.environ["HF_TOKEN"],
        config=config,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
        pad_token=DEFAULT_PAD_TOKEN,
        use_auth_token=os.environ["HF_TOKEN"],
        trust_remote_code=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    return config, model, tokenizer


def log_stats(pbar, wandb, epoch, loss_tensor, grad_norm, scheduler):
    last_lr = scheduler.get_last_lr()[0]

    wandb.log(
        {
            "current_loss": loss_tensor,
            "current_epoch": epoch,
            "learning_rate": last_lr,
            "grad_norm": grad_norm,
        },
    )

    # for consistent logging
    current_loss = f"{loss_tensor:.4f}"
    current_lr = f"{last_lr:.10f}"

    pbar.set_description(f"Epoch {epoch:.2f}, Loss: {current_loss}, LR: {current_lr}")


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    model_name = "bigcode/starcoderbase-1b"
    seed = 89691  # adjust your seed
    transformers.set_seed(seed)

    run_id = str(uuid.uuid4())
    output_dir = f"./outputs/{model_name}/{run_id}"
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I_%M_%S_%p")
    scheduler_type = "cosine"
    max_length = 2048
    disable_dropout = False
    gradient_checkpointing = True
    clip_gradients = True
    shuffle = True
    batch_size = 2  # adjust to whatever your GPU supports, I used 14 for 3090s
    epochs = 10
    acc_steps = 0  # TODO: implement grad acc
    lr = 7e-05
    weight_decay = 0.0
    gradient_clipping = 1.0

    model_config, model, tokenizer = setup_model(model_name, max_length)
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            GPTBigCodeBlock,
        },
    )

    fsdp_config = dict(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        device_id=torch.cuda.current_device(),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # will use slightly more memory vs. no prefetch
        param_init_fn=None,
        cpu_offload=None,
    )

    # wrap model with FSDP
    model = FSDP(model, **fsdp_config)
    optimizer = get_optimizer(model, lr, weight_decay)

    train_ds = [
        "data/train.jsonl"
    ]  # put your data in data folder, you can pass a list here of files
    val_ds = [
        "data/validation.jsonl"
    ]  # put your data in data folder, you can pass a list here of files
    train_dataset = SupervisedDataset(tokenizer, train_ds)
    val_dataset = SupervisedDataset(tokenizer, val_ds)
    collator = DataCollatorForSupervisedDataset(tokenizer)

    train_sampler, train_loader, val_loader = get_dataloaders(
        train_dataset,
        val_dataset,
        world_size,
        local_rank,
        shuffle,
        seed,
        collator,
        batch_size,
    )

    total_steps_per_epoch = len(train_loader)
    max_steps = total_steps_per_epoch * epochs

    scheduler = get_scheduler(local_rank, scheduler_type, optimizer, max_steps)

    if local_rank == 0:
        run = wandb.init(
            project="starcoder-1b",
            name=run_id,
            config={
                "model_name": model_name,
                "run_id": run_id,
                "date": date_of_run,
                "dataset_size": len(train_dataset),
                "dataset": " ".join(train_ds),
                "validation": " ".join(val_ds),
                "weight_decay": weight_decay,
                "clip_gradients": clip_gradients,
                "learning_rate": lr,
                "shuffle": shuffle,
                "seed": seed,
                "disable_dropout": disable_dropout,
                "epochs": epochs,
                "acc_steps": acc_steps,
                "batch_size": batch_size,
                "total_batch_size": batch_size * world_size,
                "scheduler_type": scheduler_type,
            },
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if disable_dropout:
        disable_model_dropout(model)

    dist.barrier()
    model.train()

    for epoch in range(0, epochs):
        train_sampler.set_epoch(epoch)
        current_epoch = epoch + 1

        pbar = tqdm(
            enumerate(train_loader),
            total=total_steps_per_epoch,
            colour="blue",
            desc=f"Epoch {epoch}.00",
            disable=(local_rank != 0),
        )

        for step, batch in pbar:
            current_step = step + 1

            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "labels": batch["labels"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
            }

            # forward
            outputs = model(**inputs)
            loss = outputs.loss

            # backward
            loss.backward()

            # clipping
            if clip_gradients:
                grad_norm = clip_model_gradients(model, gradient_clipping)

            # weight update
            optimizer.step()
            scheduler.step()

            # zero gradients after weight update
            optimizer.zero_grad(set_to_none=True)

            # detach from graph
            loss = loss.detach()

            # avg loss over all processes
            loss = get_all_reduce_mean(loss).item()

            if local_rank == 0:
                log_epoch = round((current_step / total_steps_per_epoch), 2) + epoch
                log_stats(
                    pbar,
                    wandb,
                    log_epoch,
                    loss,
                    grad_norm,
                    scheduler,
                )

    save_model(local_rank, model, tokenizer, output_dir)
