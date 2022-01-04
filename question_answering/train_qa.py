#!/usr/bin/env python
# coding=utf-8


import logging
import math
import os
import random
import eval_qa
import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric
from dataset import QuestionAnswerDataset
from torch.utils.data.dataloader import DataLoader
from argparse import ArgumentParser, Namespace
from tqdm import trange, tqdm
from pathlib import Path
import transformers
from accelerate import Accelerator
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
    set_seed,
)
from utils_qa import postprocess_qa_predictions
from torch.utils.tensorboard import SummaryWriter


def main(args):

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

    # preprocess
    train_val_data_path = { "train": str(args.data_dir / 'train.json'),
                            "validation": str(args.data_dir / 'public.json')}
    predict_data_path = {"predict": str(args.data_dir / 'private.json')}
    raw_data = {}
    # for key, path in data_path.items():
    #     print(f"Loading {key} data...")
    #     raw_data[key] = json.loads(path.read_text())
    
    raw_datasets = load_dataset('json', data_files=train_val_data_path, field = "data", cache_dir = str(args.cache_dir))
    predict_dataset = load_dataset('json', data_files=predict_data_path, field = "data", cache_dir = str(args.cache_dir))
    raw_datasets['predict'] = predict_dataset['predict']
    # config = AutoConfig.from_pretrained(args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast = True)
    
    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.

    column_names = raw_datasets["train"].column_names
    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples['question'],
            examples['context'],
            truncation="only_second",
            max_length=args.train_max_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples['answers'][sample_index] # There are many answers per qa, choose the first one
            # If no answers are given, set the cls_index as answer.
            if len(answers) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers[0]["start"]
                end_char = start_char + len(answers[0]["text"])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
    # Training preprocessing
    train_dataset = raw_datasets["train"]
    # prepare_train_features(train_dataset[:100])
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=8,
        remove_columns=column_names,
        load_from_cache_file=True,
    )

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples['question'],
            examples['context'],
            truncation="only_second",
            max_length=args.eval_max_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    eval_examples = raw_datasets["validation"]
    prepare_validation_features(eval_examples)
    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=8,
        remove_columns=column_names,
        load_from_cache_file=True,
    )
    eval_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])
    column_names = ['id', 'question', 'context']
    if args.do_predict:
        predict_examples = raw_datasets['predict']
        predict_dataset = predict_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=8,
            remove_columns=column_names,
            load_from_cache_file=True,
        )
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    data_collator = default_data_collator
    # data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)

    if args.do_predict:
        predict_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])
        predict_dataloader = DataLoader(
            predict_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size
        )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.predict_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if stage == 'eval':
            formatted_predictions = {k: v for k, v in predictions.items()}
            references = {ex["id"]: {'answers': [sample['text'] for sample in ex['answers']]} for ex in examples}
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)
        elif stage == 'predict':
            formatted_predictions = {k: v for k, v in predictions.items()}
            return formatted_predictions

    # metric = load_metric("f1")

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    if args.load_model == True:
        print(f"Load model from check point: \"{args.checkpoint_dir}\"")
        model = AutoModelForQuestionAnswering.from_pretrained(args.checkpoint_dir)
    else:
        print(f"Load model from {args.pretrained_model}")
        model = AutoModelForQuestionAnswering.from_pretrained(args.pretrained_model)
    device = accelerator.device
    model = model.to(device)


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
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, predict_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, predict_dataloader
    )


    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                              patience = 10,
                                                              mode = 'min',
                                                              cooldown = 10,
                                                              verbose = True)

    # Train!
    gradient_accumulation_steps = args.ideal_batch_size / args.batch_size
    total_batch_size = args.ideal_batch_size * args.num_epochs
    num_of_eval_per_epoch = math.floor(args.ratio_of_eval * len(eval_dataset))
    # writer = SummaryWriter(args.output_dir)


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")

    # Only show the progress bar once on each machine.
    for epoch in trange(args.num_epochs, desc="Epoch"):
        completed_steps = 0
        model.train()
        epoch_train_loss = 0
        epoch_eval_loss = 0
        pbar = tqdm(total = args.num_train_steps_per_epoch)

        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            train_loss = float(outputs.loss)
            epoch_train_loss += train_loss
            # train_loss = train_loss / gradient_accumulation_steps
            accelerator.backward(outputs.loss)

            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                completed_steps += 1
                pbar.update(1)
                if completed_steps >= args.num_train_steps_per_epoch:
                    break
        
        model.eval()
        # Validation
        all_start_logits = []
        all_end_logits = []
        eval_steps = 0
        bar = tqdm(total = num_of_eval_per_epoch)
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

                eval_steps += 1
                bar.update(1)
                if eval_steps > num_of_eval_per_epoch:
                    break

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        eval_dataset.set_format(type=None, columns=list(eval_dataset.features.keys()))
        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
        eval_metric = eval_qa.compute_metrics(answers = prediction.label_ids,predictions = prediction.predictions)
        eval_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])
        del start_logits_concat
        del end_logits_concat
        del outputs_numpy
        del prediction
        writer.add_scalar('Train Loss', epoch_train_loss, epoch)
        writer.add_scalar('Eval F1', eval_metric['f1'], epoch)
        writer.add_scalar('Eval EM', eval_metric['em'], epoch)
        lr_scheduler.step(epoch_train_loss)
    # eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        logger.info(f"Epoch {epoch}: Train Loss {epoch_train_loss}, Evaluation metrics: {eval_metric}")
        if epoch > 0:
            output_path = args.output_dir / f"ckpt_{epoch}"
            output_path.mkdir()
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            print(f"Reach checkpoint {epoch} and save model in {output_path}")
            unwrapped_model.save_pretrained(output_path)

    # writer.close()
    # Prediction
    if args.do_predict:
        print("Do some predict")
        all_start_logits = []
        all_end_logits = []
        for step, batch in enumerate(predict_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        # Now we need to add extra columns which we removed for post processing
        predict_dataset.set_format(type=None, columns=list(predict_dataset.features.keys()))
        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy, stage = 'predict')

    # if args.output_dir is not None:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

def parse_args(paras) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=paras['data_dir'])
    parser.add_argument("--cache_dir", type=Path, default=paras['cache_dir'])
    parser.add_argument("--output_dir", type=Path, default=paras['output_dir'])
    parser.add_argument("--predict_dir",  type=Path, default=paras['predict_dir'])
    parser.add_argument("--load_model", action="store_true", default=paras['load_model'])
    parser.add_argument("--checkpoint_dir", type=Path, default=paras['checkpoint_dir'])
    parser.add_argument("--pretrained_model", type=str, default=paras['pretrained_model'])
    parser.add_argument("--train_max_length", type=int, default=paras['train_max_length'])
    parser.add_argument("--eval_max_length", type=int, default=paras['eval_max_length'])
    parser.add_argument("--doc_stride", type=int, default=paras['doc_stride'])
    parser.add_argument("--lr", type=float, default=paras['lr'])
    parser.add_argument("--weight_decay", type=float, default=paras['weight_decay'])
    parser.add_argument("--batch_size", type=int, default=paras['batch_size'])
    parser.add_argument("--ideal_batch_size", type=int, default=paras['ideal_batch_size'])
    parser.add_argument("--eval_batch_size", type=int, default=paras['eval_batch_size'])
    parser.add_argument("--num_epochs", type=int, default=paras['num_epochs'])
    parser.add_argument("--num_train_steps_per_epoch", type=int, default=paras['num_train_steps_per_epoch'])
    parser.add_argument("--ratio_of_eval", type=int, default=paras['ratio_of_eval'])
    parser.add_argument("--device", type=torch.device, default=paras['device'])
    parser.add_argument("--do_predict", action="store_true", default=paras['do_predict'])
    parser.add_argument("--n_best_size", type=int, default=paras['n_best_size'])
    parser.add_argument("--max_answer_length", type=int, default=paras['max_answer_length'])
    parser.add_argument("--null_score_diff_threshold", type=float, default=paras['null_score_diff_threshold'])
    args = parser.parse_args()

    if not args.output_dir.exists():
        args.output_dir.mkdir()        
    if not args.predict_dir.exists():
        args.predict_dir.mkdir()        
    return args

if __name__ == "__main__":
    config = {
        'data_dir': './dataset/qa',
        'cache_dir': './cache/qa',
        'output_dir': './model/qa',
        'predict_dir': './predict',
        'load_model': False,
        'checkpoint_dir': './model/qa',
        'pretrained_model': 'hfl/chinese-roberta-wwm-ext',
        'train_max_length': 512,
        'eval_max_length': 512,
        'doc_stride': 128,
        'lr': 1e-5,
        'weight_decay': 2e-2,
        'batch_size': 4,
        'ideal_batch_size': 64,
        'eval_batch_size': 8,
        'num_epochs': 0,
        'num_train_steps_per_epoch': 1000,
        'ratio_of_eval': 1,
        'device': 'cuda',
        'do_predict': True,
        'n_best_size': 20,
        'max_answer_length': 30,
        'null_score_diff_threshold': 0.0
    }
    args = parse_args(config)
    main(args)
