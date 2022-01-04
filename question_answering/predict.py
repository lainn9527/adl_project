import torch
import json
import math
import pickle
from dataset import MultipleChoiceDataset
from preprocess import preprocess_multiple_choice, build_question_answer_data, postprocess_qa_predictions, create_and_fill_np_array
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMultipleChoice, AutoModelForQuestionAnswering, AutoTokenizer, default_data_collator
from accelerate import Accelerator
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from tqdm import trange, tqdm
from datetime import datetime
from datasets import load_dataset

def predict(args):
    print(f'Load tokenizer from {args.mc_pretrained_model}')
    tokenizer = AutoTokenizer.from_pretrained(args.mc_pretrained_model, use_fast = True)
    context_path = args.context_path
    context = json.loads(context_path.read_text()) # list of strings
    print(f"Load mc model from check point: \"{args.mc_model_path}\"")
    mc_model = AutoModelForMultipleChoice.from_pretrained(args.mc_model_path)
    accelerator = Accelerator()
    device = accelerator.device
    mc_model = mc_model.to(device)

    predict_data_path = args.test_data_path
    predict_data = json.loads(predict_data_path.read_text()) # list of dict
    encoded_predict_path = args.cache_dir / 'private.pkl'
    print("Preprocess multiple choice data...")
    encoded_predict = preprocess_multiple_choice(tokenizer, predict_data, context, 512, encoded_predict_path)

    predict_dataset = MultipleChoiceDataset(encoded_predict, tokenizer)
    predict_dataloader = DataLoader(predict_dataset, batch_size = 1, collate_fn = predict_dataset.eval_collate_fn)

    mc_model, predict_dataloader = accelerator.prepare(mc_model, predict_dataloader)
    # context_ids = json.loads(Path('predict/mc_roberta/context_id.json').read_text())
    context_ids = []
    print("Predict relevant context...")
    pbar = tqdm(total = len(predict_dataset))
    for i, batch in enumerate(predict_dataloader):
        with torch.no_grad():
            output = mc_model(**batch)
            prediction = output.logits.argmax(dim=-1)[0].item()
        context_id = predict_data[i]['paragraphs'][prediction]
        context_ids.append(context_id)
        pbar.update(1)
    
    # make qa dataset
    print('Preprocess question answer dataset')
    raw_data = json.loads(predict_data_path.read_text()) # list of dict
    context = json.loads(context_path.read_text()) # list of strings
    qa_data_path = args.cache_dir / 'qa_test.json'
    build_question_answer_data(raw_data, context, output_path = qa_data_path, context_id = context_ids)

    # predict qa
    print(f'Load tokenizer from {args.qa_pretrained_model}')
    tokenizer = AutoTokenizer.from_pretrained(args.qa_pretrained_model, use_fast = True)
    predict_data_path = {"predict": str(qa_data_path)}
    predict_dataset = load_dataset('json', data_files=predict_data_path, field = "data", cache_dir = str(args.cache_dir))
    predict_examples = predict_dataset['predict']

    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples['question'],
            examples['context'],
            truncation="only_second",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
        
    predict_dataset = predict_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=8,
        remove_columns=['id', 'question', 'context'],
        load_from_cache_file=True,
    )
    predict_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])
    data_collator = default_data_collator
    predict_dataloader = DataLoader(predict_dataset, collate_fn=data_collator, batch_size=8)
    print(f"Load qa model from check point: \"{args.qa_model_path}\"")
    qa_model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model_path)
    qa_model = qa_model.to(device)
    qa_model, predict_dataloader = accelerator.prepare(qa_model, predict_dataloader)

    print("Predict QA")
    all_start_logits = []
    all_end_logits = []
    pbar = tqdm(total = len(predict_dataset)) 
    for step, batch in enumerate(predict_dataloader):
        with torch.no_grad():
            outputs = qa_model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())
        pbar.update(1)

    print('Postprocess QA')
    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
    start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)
    # Now we need to add extra columns which we removed for post processing
    predict_dataset.set_format(type=None, columns=list(predict_dataset.features.keys()))
    outputs_numpy = (start_logits_concat, end_logits_concat)
    predictions = postprocess_qa_predictions(
        examples=predict_examples,
        features=predict_dataset,
        predictions=outputs_numpy,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        output_path=args.output_path,
    )
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--context_path", type=Path)
    parser.add_argument("--test_data_path", type=Path)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--mc_model_path", type=Path, default = 'model/mc')
    parser.add_argument("--qa_model_path", type=Path, default = 'model/qa')
    parser.add_argument("--cache_dir", type=Path, default='./cache')
    parser.add_argument("--mc_pretrained_model", type=str, default='bert-base-chinese')
    parser.add_argument("--qa_pretrained_model", type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument("--device", type=torch.device, default='cuda')
    args = parser.parse_args()

    if not args.cache_dir.exists():
        args.cache_dir.mkdir()
    return args

if __name__ == "__main__":
    args = parse_args()
    predict(args)
    # python3.8 predict.py --context_path ./dataset/context.json --test_data_path ./dataset/private.json --output_path ./predict.json