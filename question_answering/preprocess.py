import json
import pickle
import collections
import numpy as np
from typing import Optional, Tuple
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

def preprocess_multiple_choice(tokenizer, items, context, max_length, output_path):
    encoded_items = []
    predict = False if 'relevant' in items[0].keys() else True

    for item in tqdm(items):
        text_paragraphs = [context[i] for i in item['paragraphs']]
        questions = [item['question']] * len(text_paragraphs)
        encoded_item = tokenizer(
            questions, text_paragraphs, max_length=max_length, truncation='only_second')
        if not predict:
            for i, _id in enumerate(item['paragraphs']):
                if _id == item['relevant']:
                    encoded_item['label'] = i
                    break
        encoded_items.append(encoded_item)

    with open(output_path, 'wb') as fp:
        pickle.dump(encoded_items, fp)

    return encoded_items


def build_question_answer_data(items, context, output_path, context_id=None):
    context_ids = [item['relevant']
                   for item in items] if context_id == None else context_id
    predict = False if 'answers' in items[0].keys() else True
    for i, item in enumerate(items):
        item.pop('paragraphs')
        if not predict:
            item.pop('relevant')
        item['context'] = context[context_ids[i]]

    with open(output_path, 'w') as fp:
        json.dump({'data': items}, fp, ensure_ascii=False, indent=2)

    return {'data': items}


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

# Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor


def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    step = 0
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]
        if step + batch_size < len(dataset):
            logits_concat[step: step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]
        step += batch_size
    return logits_concat


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_path: Optional[str] = None,
    is_world_process_zero: bool = True,
):
    assert len(
        predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions
    assert len(predictions[0]) == len(
        features), f"Got {len(predictions[0])} predictions and {len(features)} features."

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(
            i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_prediction = None
        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None)
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(
                start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(
                end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[
            :n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - \
                best_non_null_pred["start_logit"] - \
                best_non_null_pred["end_logit"]
            # To be JSON-serializable.
            scores_diff_json[example["id"]] = float(score_diff)
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v)
             for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.

    with open(output_path, "w") as writer:
        writer.write(json.dumps(all_predictions,
                                indent=4, ensure_ascii=False) + "\n")

    return all_predictions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path('./dataset'))
    args = parser.parse_args()
    data_dir = args.data_dir
    model_name = 'bert-base-chinese'
    bert_max_length = 512
    context_path = data_dir / 'context.json'
    context = json.loads(context_path.read_text())  # list of strings

    cache_dir = Path('./cache')
    data_types = ['train', 'public', 'private']

    # # preprocess for multiple answer
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
    # mc_cache_dir = cache_dir / 'mc'
    # print('Preprocess multiple answer')
    # for data_type in data_types:
    #   print(f"Preprocess {data_type} data...")
    #   data_path = data_dir / f'{data_type}.json'
    #   raw_data = json.loads(data_path.read_text()) # list of dict
    #   encoded_data_path = mc_cache_dir / f'./{data_type}.pkl'
    #   preprocess_multiple_choice(tokenizer, raw_data, context, bert_max_length, encoded_data_path)

    # preprocess for question answer

    qa_data_dir = data_dir / 'qa'
    if not qa_data_dir.exists():
        qa_data_dir.mkdir(parents=True)
    predict_context_path = Path('./predict/context_id.json')
    print('Preprocess question answer')
    for data_type in data_types:
        print(f"Preprocess {data_type} data...")
        data_path = data_dir / f'{data_type}.json'
        raw_data = json.loads(data_path.read_text())  # list of dict
        encoded_data_path = qa_data_dir / f'./{data_type}.json'
        if data_type == 'private':
            context_ids = json.loads(predict_context_path.read_text())
            build_question_answer_data(
                raw_data, context, encoded_data_path, context_id=context_ids)
        else:
            build_question_answer_data(raw_data, context, encoded_data_path)
