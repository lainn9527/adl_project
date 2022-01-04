import argparse
import datasets
import numpy as np
import torch
import json
import jsonlines
from datasets import load_dataset, load_metric, DatasetDict
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from transformers.file_utils import is_offline_mode
from torch.utils.tensorboard import SummaryWriter

def main(args, gen_config):
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    predict_data_path = {"predict": str(args.file_path)}
    raw_datasets = load_dataset('json', data_files=predict_data_path, field = "data", cache_dir = args.cache_dir)

    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, config = config)
    # print(f'Resume from {args.resume_path}')
    tokenizer = AutoTokenizer.from_pretrained('google/mt5-small', use_fast=True)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = 'summarize: '
    text_column = 'maintext'
    padding = "max_length"

    def preprocess_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=256, padding=padding, truncation=True)
        return model_inputs

    column_names = raw_datasets["predict"].column_names
    column_names.remove('id')
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=column_names, load_from_cache_file=True
    )

    predict_dataset = processed_datasets["predict"]
    label_pad_token_id = -100
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=label_pad_token_id)
    def data_collator(batch):
        attention_masks = []
        input_ids = []
        ids = []
        for item in batch:
            attention_masks.append(item['attention_mask'])
            input_ids.append(item['input_ids'])
            ids.append(item['id'])
            
        
        return {'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_masks),
                'ids': ids}
    predict_dataloader = DataLoader(predict_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    model, predict_dataloader = accelerator.prepare(model, predict_dataloader)
    model.eval()
    preds = []
    for step, batch in enumerate(predict_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_config,
            )

            generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True)

            for pred, _id in zip(decoded_preds, batch['ids']):
                preds.append({'title': pred, 'id': _id})
    
    with jsonlines.open(args.output_path, 'w') as writer:
        writer.write_all(preds)
        

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    parser.add_argument("--file_path", type=Path)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--model_path", type=Path, default = Path('./model/'))
    parser.add_argument("--cache_dir", type=Path, default = Path('./cache/'))
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    args = parser.parse_args()
    return args
            


if __name__ == "__main__":
    gen_config = {
        "max_length": 64,
        "num_beams": 24,
        "top_k": 100,
        "temperature": 1.5
    }
    args = parse_args()

    # preprocess
    l = []
    with jsonlines.open(args.file_path) as reader:
        for obj in reader:
            l.append(obj)
    args.cache_dir.mkdir(exist_ok = True)
    preprocess_path = args.cache_dir / 'data.json'
    args.file_path = preprocess_path
    with open(args.file_path, 'w') as fp:
        json.dump({'data': l}, fp, indent=4, ensure_ascii=False)
    main(args, gen_config)
