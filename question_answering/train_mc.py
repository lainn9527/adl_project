import torch
import json
import math
import pickle
from dataset import MultipleChoiceDataset
from datasets import load_metric
from preprocess import preprocess_multiple_choice
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer, AdamW
from accelerate import Accelerator
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from tqdm import trange
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def train_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast = True)
    context_path = args.data_dir / 'context.json'
    context = json.loads(context_path.read_text()) # list of strings
    encoded_train_path = args.cache_dir / 'train.pkl'
    encoded_eval_path = args.cache_dir / 'public.pkl'

    if args.cache_dir.exists():
        print("There are preprocessed data\nLoading training data...")
        encoded_train = pickle.loads(encoded_train_path.read_bytes())
        print("Loading evaluating data...")
        encoded_eval = pickle.loads(encoded_eval_path.read_bytes())
    else:
        print("There is no preprocessed data")
        args.cache_dir.mkdir(parents = True)
        train_path = args.data_dir / 'train.json'
        eval_path = args.data_dir / 'public.json'
        train_data = json.loads(train_path.read_text()) # list of dict
        eval_data = json.loads(eval_path.read_text()) # list of dict
        print("Preprocess training data...")
        encoded_train = preprocess_multiple_choice(tokenizer, train_data, context, args.train_max_length, encoded_train_path)
        print("Preprocess evaluating data...")
        encoded_eval = preprocess_multiple_choice(tokenizer, eval_data, context, args.eval_max_length, encoded_eval_path)
    
    train_dataset = MultipleChoiceDataset(encoded_train, tokenizer)
    eval_dataset = MultipleChoiceDataset(encoded_eval, tokenizer)
    num_of_eval_per_epoch = math.floor(args.ratio_of_eval * len(eval_dataset))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size = args.batch_size,
                                  num_workers = 8,
                                  shuffle = True,
                                  collate_fn = train_dataset.train_collate_fn)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size = 1,
                                 shuffle = True,
                                 collate_fn = eval_dataset.eval_collate_fn)

    # for epoch in epoch_bar:
    #     for step, batch in enumerate(train_dataloader):
    #         print(step)
    if args.checkpoint_dir.exists():
        print(f"Load model from check point: \"{args.checkpoint_dir}\"")
        model = AutoModelForMultipleChoice.from_pretrained(args.checkpoint_dir)
    else:
        print(f"Load model from {args.pretrained_model}")
        model = AutoModelForMultipleChoice.from_pretrained(args.pretrained_model)
    accelerator = Accelerator()
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
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    gradient_accumulation_steps = args.ideal_batch_size / args.batch_size
    

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                              patience = 10,
                                                              mode = 'min',
                                                              cooldown = 10,
                                                              verbose = True)

    metric = load_metric("accuracy")
    writer = SummaryWriter(args.checkpoint_dir)
    for epoch in trange(args.num_epoch, desc="Epoch"):
        completed_steps = 0
        model.train()
        epoch_train_loss = 0
        epoch_eval_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            train_loss = outputs.loss
            epoch_train_loss += float(train_loss)
            # train_loss = train_loss / gradient_accumulation_steps
            accelerator.backward(train_loss)

            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                completed_steps += 1

                if completed_steps >= args.num_train_steps_per_epoch:
                    break

        train_metric = metric.compute()

        model.eval()
        num_of_eval_samples = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                eval_loss = outputs.loss
                epoch_eval_loss += float(eval_loss)

            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
            # num_of_eval_samples += 1
            if num_of_eval_samples > num_of_eval_per_epoch:
                break
        
        eval_metric = metric.compute()
        train_metric = eval_metric
        accelerator.print(f"epoch {epoch}")
        accelerator.print(f"train_loss: {epoch_train_loss}, train_eval: {train_metric['accuracy']}\neval_loss: {epoch_eval_loss} eval: {eval_metric['accuracy']}")
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/eval', epoch_eval_loss, epoch)
        writer.add_scalar('Accuracy/train', train_metric['accuracy'], epoch)
        writer.add_scalar('Accuracy/eval', eval_metric['accuracy'], epoch)
        lr_scheduler.step(train_loss)

        # checkpoint
        if epoch > 0:
            output_path = args.output_dir / f"ckpt_{epoch}"
            output_path.mkdir()
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            print(f"Reach checkpoint {epoch} and save model in {output_path}")
            unwrapped_model.save_pretrained(output_path)

    writer.close()

    if not args.output_dir.exists():
        args.output_dir.mkdir()
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    print(f"Finish training and save model in {args.output_dir}")
    unwrapped_model.save_pretrained(args.output_dir)


def predict_mc(args):
    print(f'Load tokenizer from {args.pretrained_model}')
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast = True)
    context_path = args.data_dir / 'context.json'
    context = json.loads(context_path.read_text()) # list of strings

    if args.checkpoint_dir.exists():
        print(f"Load model from check point: \"{args.checkpoint_dir}\"")
        model = AutoModelForMultipleChoice.from_pretrained(args.checkpoint_dir)
    else:
        print(f"Load model from {args.pretrained_model}")
        model = AutoModelForMultipleChoice.from_pretrained(args.pretrained_model)
    accelerator = Accelerator()
    device = accelerator.device
    model = model.to(device)

    predict_data_path = args.data_dir / 'private.json'
    predict_data = json.loads(predict_data_path.read_text()) # list of dict
    encoded_predict_path = args.cache_dir / 'private.pkl'
    print('Load predict data...')
    if args.cache_dir.exists():
        encoded_predict = pickle.loads(encoded_predict_path.read_bytes())
    else:
        encoded_predict = preprocess_multiple_choice(tokenizer, predict_data, context, args.eval_max_length, encoded_predict_path)

    predict_dataset = MultipleChoiceDataset(encoded_predict, tokenizer)
    predict_dataloader = DataLoader(predict_dataset,
                                    batch_size = 1,
                                    collate_fn = predict_dataset.eval_collate_fn)
    model, predict_dataloader = accelerator.prepare(model, predict_dataloader)
    # output context id
    context_ids = []
    for i, batch in enumerate(predict_dataloader):
        with torch.no_grad():
            output = model(**batch)
            prediction = output.logits.argmax(dim=-1)[0].item()
        context_id = predict_data[i]['paragraphs'][prediction]
        context_ids.append(context_id)
    
    if not args.predict_dir.exists():
        args.predict_dir.mkdir()
    predict_path = args.predict_dir / 'context_id.json'
    print(f'Saving context_ids in {predict_path}')
    with open(predict_path, 'w') as fp:
        json.dump(context_ids, fp)

    
    

    

def parse_args(paras) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=paras['data_dir'])
    parser.add_argument("--cache_dir", type=Path, default=paras['cache_dir'])
    parser.add_argument("--output_dir", type=Path, default=paras['output_dir'])
    parser.add_argument("--predict_dir",  type=Path, default=paras['predict_dir'])
    parser.add_argument("--checkpoint_dir", type=Path, default=paras['checkpoint_dir'])
    parser.add_argument("--pretrained_model", type=str, default=paras['pretrained_model'])
    parser.add_argument("--train_max_length", type=int, default=paras['train_max_length'])
    parser.add_argument("--eval_max_length", type=int, default=paras['eval_max_length'])
    parser.add_argument("--lr", type=float, default=paras['lr'])
    parser.add_argument("--weight_decay", type=float, default=paras['weight_decay'])
    parser.add_argument("--batch_size", type=int, default=paras['batch_size'])
    parser.add_argument("--ideal_batch_size", type=int, default=paras['ideal_batch_size'])
    parser.add_argument("--num_epoch", type=int, default=paras['num_epoch'])
    parser.add_argument("--num_train_steps_per_epoch", type=int, default=paras['num_train_steps_per_epoch'])
    parser.add_argument("--ratio_of_eval", type=int, default=paras['ratio_of_eval'])
    parser.add_argument("--device", type=torch.device, default=paras['device'])

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    config = {
        'data_dir': './dataset',
        'cache_dir': './cache/mc',
        'output_dir': './model/mc',
        'predict_dir': './predict',
        'checkpoint_dir': './model/mc',
        'pretrained_model': 'bert-base-chinese',
        'train_max_length': 256,
        'eval_max_length': 512,
        'lr': 1e-4,
        'weight_decay': 2e-2,
        'batch_size': 4,
        'ideal_batch_size': 64,
        'num_epoch': 25,
        'num_train_steps_per_epoch': 25,
        'ratio_of_eval': 0.1,
        'device': 'cuda'
    }
    args = parse_args(config)
    train_model(args)
    predict_mc(args)