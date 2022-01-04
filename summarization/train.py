import argparse
import logging
import math
import os
import random

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric, DatasetDict
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from torch.utils.tensorboard import SummaryWriter

# from tw_rouge import get_rouge
logger = logging.getLogger(__name__)

# try:
#     nltk.data.find("tokenizers/punkt")
# except (LookupError, OSError):
#     if is_offline_mode():
#         raise LookupError(
#             "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
#         )
#     with FileLock(".lock") as lock:
#         nltk.download("punkt", quiet=True)


def parse_args(config):
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
    )
    parser.add_argument("--train_file", type=str, default=config["train_file"])
    parser.add_argument("--validation_file", type=str, default=config["validation_file"])
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=config["max_source_length"],
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default="summarize: ",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
    )
    parser.add_argument("--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets")
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=config["max_target_length"],
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=config["val_max_target_length"],
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=config["max_length"],
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=config["model_name_or_path"],
    )
    parser.add_argument("--resume", action="store_true", default=config["resume"])
    parser.add_argument("--resume_path", type=str, default=config["resume_path"])
    parser.add_argument("--resume_epoch", type=int, default=config["resume_epoch"])
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="maintext",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default="title",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=config["per_device_train_batch_size"],
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--ideal_batch_size",
        type=int,
        default=config["ideal_batch_size"],
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=config["per_device_eval_batch_size"],
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config["learning_rate"],
    )
    parser.add_argument("--weight_decay", type=float, default=config["weight_decay"], help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=config["num_train_epochs"], help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=config["max_train_steps"],
    )
    parser.add_argument(
        "--max_eval_nums",
        type=int,
        default=config["max_eval_nums"],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lr_scheduler_type",
        # type=SchedulerType,
        default=config["lr_scheduler_type"],
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=config["num_warmup_steps"], help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=config["output_dir"], help="Where to store the final model.")
    parser.add_argument("--cache_dir", type=str, default=config["cache_dir"], help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main(config, gen_config):
    args = parse_args(config)
    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, field="data", cache_dir=args.cache_dir)

    if args.resume:
        config = AutoConfig.from_pretrained(args.resume_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.resume_path,
            config=config,
        )
        print(f"Resume from {args.resume_path}")

    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
        print(f"Train on {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix
    column_names = raw_datasets["train"].column_names
    text_column = args.text_column
    summary_column = args.summary_column

    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=column_names, load_from_cache_file=True)

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    args.gradient_accumulation_steps = args.ideal_batch_size / args.per_device_train_batch_size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    from tw_rouge import get_rouge

    num_total_update = args.num_train_epochs * args.max_train_steps

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_total_update,
    )

    # Metric
    writer = SummaryWriter(args.output_dir)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.

    base_epoch = 0 if not args.resume else args.resume_epoch
    for epoch in range(base_epoch, base_epoch + args.num_train_epochs):
        print(f"Epoch {epoch} start")
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        model.train()
        training_loss = 0
        completed_steps = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            training_loss += float(loss)
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        preds = []
        refs = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_config,
                )

                generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                preds += decoded_preds
                refs += decoded_labels
                if len(preds) >= args.max_eval_nums:
                    break

        result = get_rouge(preds, refs)
        result = {k: {k2: round(v2, 4) for k2, v2 in v.items()} for k, v in result.items()}
        writer.add_scalar("rouge-1", result["rouge-1"]["f"] * 100, epoch)
        writer.add_scalar("rouge-2", result["rouge-2"]["f"] * 100, epoch)
        writer.add_scalar("rouge-l", result["rouge-l"]["f"] * 100, epoch)

        print(f"\nrouge-1: {result['rouge-1']['f'] * 100}, rouge-2: {result['rouge-2']['f'] * 100}, rouge-l: {result['rouge-l']['f'] * 100}")
        if epoch > 0:
            output_path = f"{args.output_dir}/ckpt_{epoch}"
            os.makedirs(output_path, exist_ok=True)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_path, save_function=accelerator.save)

    writer.close()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    data_path = Path("data_json")
    output_dir = Path("./result")
    cache_dir = Path("./cache")

    config = {
        "train_file": str(data_path / "train.json"),
        "validation_file": str(data_path / "public.json"),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "max_source_length": 256,
        "max_target_length": 64,
        "val_max_target_length": 64,
        "max_length": 128,
        "model_name_or_path": "google/mt5-small",
        "resume": False,
        "resume_path": str("./result/le-3_64_24_20_top100_t1.5/ckpt_3"),
        "resume_epoch": 4,
        "per_device_train_batch_size": 2,
        "ideal_batch_size": 64,
        "per_device_eval_batch_size": 1,
        "num_train_epochs": 30,
        "max_train_steps": 1000,
        "max_eval_nums": 500,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "num_warmup_steps": 3,
    }
    gen_config = {"max_length": 64, "num_beams": 24, "top_k": 100, "temperature": 1.5}
    main(config, gen_config)
